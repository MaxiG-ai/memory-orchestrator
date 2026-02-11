import litellm
import weave
import os
import time

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union, Iterable
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletion,
)
from memorch.utils.config import load_configs, ExperimentConfig, ModelDef
from memorch.memory_processing import MemoryProcessor
from memorch.utils.token_count import get_token_count
from memorch.utils.logger import get_logger

logger = get_logger("Orchestrator")


def _serialize_message(msg: Any) -> Dict:
    """Convert a message to a JSON-serializable dictionary.

    Handles raw dicts, pydantic models, and OpenAI SDK objects like
    ChatCompletionMessageToolCall.
    """
    if isinstance(msg, dict):
        # Recursively serialize nested objects
        result = {}
        for key, value in msg.items():
            if hasattr(value, "model_dump"):
                result[key] = value.model_dump()
            elif isinstance(value, list):
                result[key] = [
                    v.model_dump() if hasattr(v, "model_dump") else v for v in value
                ]
            else:
                result[key] = value
        return result
    elif hasattr(msg, "model_dump"):
        return msg.model_dump()
    else:
        return msg


@dataclass
class CompressionMetadata:
    """Metadata about memory compression applied during request processing.

    Contains metrics useful for debugging and evaluating memory strategy performance.
    """

    input_token_count: int = 0
    compressed_token_count: int = 0
    compression_ratio: float = 1.0
    memory_method: str = ""
    processing_time_ms: float = 0.0
    loop_detected: bool = False
    strategy_metadata: Dict[str, Any] = field(default_factory=dict)
    compressed_messages: Optional[List[Dict]] = None
    original_messages: Optional[List[Dict]] = None


@dataclass
class CompressedTraceEntry:
    """A single entry in the compressed trace buffer.

    Captures the memory-processed messages for each LLM call within a session.
    """

    step: int  # Call number within session (1-indexed)
    input_token_count: int
    compressed_token_count: int
    compression_ratio: float
    memory_method: str
    compressed_messages: List[Dict]  # The actual processed messages sent to LLM


class LLMOrchestrator:
    """
    Centralized LLM interaction manager.

    Integrates:
    - LiteLLM for model interactions
    - Memory processing for context optimization
    - Comprehensive tracking with weave/wandb

    Usage:
        # Initialize
        orchestrator = LLMOrchestrator()
        orchestrator.set_active_context("gpt-5", "memory_bank")

        # Direct use
        response = orchestrator.generate(messages, tools=tools)

        # Benchmark integration (inject into model)
        runner = SAPGPTRunner(
            model_name="gpt-5",
            args=args,
            logger=logger,
            orchestrator=orchestrator  # Memory processing automatically applied
        )
    """

    def __init__(self, exp_path="config.toml", model_path="model_config.toml"):
        """
        Initialize orchestrator with configuration.

        Args:
            exp_path: Path to experiment config file
            model_path: Path to model registry config file
        """
        # Load static config
        self.cfg: ExperimentConfig = load_configs(exp_path, model_path)

        # Initialize memory processor
        self.memory_processor = MemoryProcessor(self.cfg)

        # State variables (mutable)
        self.active_model_key: str = self.cfg.enabled_models[0]
        self.active_memory_key: str = self.cfg.enabled_memory_methods[0]

        # Session-level trace buffer for compressed messages
        # Collects memory-processed messages for each LLM call within a session
        self._compressed_trace_buffer: List[CompressedTraceEntry] = []
        self._trace_step_counter: int = 0

        # Configure LiteLLM
        if self.cfg.weave_deep_logging:
            os.environ["LITELLM_LOG"] = "DEBUG"
            litellm.suppress_debug_info = False
            litellm.success_callback = ["weave", "console"]

        logger.info(f"🚀 Orchestrator initialized for: {self.cfg.experiment_name}")

    def get_exp_config(self) -> Dict[str, Any]:
        """
        Return only experiment configs (no model registry). Used for logging with weave.
        """
        exp_dict = self.cfg.model_dump()
        exp_dict.pop("model_registry", None)
        return exp_dict

    def reset_session(self):
        """
        Clear memory state. Call this before starting a new benchmark conversation.
        """
        self.memory_processor.reset_state()
        self._compressed_trace_buffer = []
        self._trace_step_counter = 0
        logger.info("🔄 Session Reset")

    def get_compressed_trace(self) -> List[CompressedTraceEntry]:
        """
        Retrieve the compressed trace buffer for the current session.

        Returns a list of CompressedTraceEntry objects, one for each LLM call
        made during this session. Use this after running a benchmark case to
        save the memory-processed messages separately.

        Returns:
            List of CompressedTraceEntry objects
        """
        return self._compressed_trace_buffer.copy()

    def get_compressed_trace_as_dicts(self) -> List[Dict]:
        """
        Retrieve the compressed trace buffer as serializable dictionaries.

        Convenience method for JSON serialization. Handles OpenAI SDK objects
        like ChatCompletionMessageToolCall by converting them to dicts.

        Returns:
            List of dictionaries representing each trace entry
        """
        return [
            {
                "step": entry.step,
                "input_token_count": entry.input_token_count,
                "compressed_token_count": entry.compressed_token_count,
                "compression_ratio": entry.compression_ratio,
                "memory_method": entry.memory_method,
                "compressed_messages": [
                    _serialize_message(msg) for msg in entry.compressed_messages
                ],
            }
            for entry in self._compressed_trace_buffer
        ]

    def set_active_context(self, model_key: str, memory_key: str):
        """
        HOTSWAP: Updates the active configuration for the next request.
        Call this inside your experiment loop to switch configurations.

        Args:
            model_key: Model identifier from model_config.toml
            memory_key: Memory strategy from config.toml

        Raises:
            ValueError: If model or memory strategy not found
        """
        if model_key not in self.cfg.model_registry:
            raise ValueError(f"Model '{model_key}' not found in registry.")
        if memory_key not in self.cfg.memory_strategies:
            raise ValueError(f"Memory strategy '{memory_key}' not defined.")

        self.active_model_key = model_key
        self.active_memory_key = memory_key

        logger.info("🔄 Context Switched")

    def get_model_config(self) -> ModelDef:
        """Helper to get current model config"""
        model_def = self.cfg.model_registry.get(self.active_model_key)
        if not model_def:
            raise ValueError(f"Model '{self.active_model_key}' not found in registry.")
        return model_def

    def get_model_kwargs_from_config(self) -> Dict[str, Any]:
        """
        Retrieve model-specific kwargs from configuration.

        Returns:
            Dictionary of model parameters (e.g., temperature) from extra fields
        """
        model_def = self.get_model_config()

        # Get all fields from the model
        all_fields = model_def.model_dump()

        # Define the base/required fields that should not be passed as kwargs
        base_fields = {
            "litellm_name",
            "context_window",
            "provider",
            "api_base",
            "api_key",
        }

        # Extract only the extra fields (kwargs)
        model_kwargs = {
            key: value
            for key, value in all_fields.items()
            if key not in base_fields and value is not None
        }

        return model_kwargs

    @weave.op(enable_code_capture=False)
    def generate_with_memory_applied(
        self,
        input_messages: List[Dict[str, str]],
        tools: Optional[List[ChatCompletionToolParam]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        return_metadata: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Any, tuple]:
        """
        Execute LLM request with memory processing and comprehensive tracking.

        This is the core method that:
        1. Logs pre-processing metrics
        2. Applies memory strategy via memory processor
        3. Executes LLM call with timing
        4. Logs post-processing metrics
        5. Handles errors with tracking

        Args:
            input_messages: Conversation messages (will be processed by memory strategy)
            tools: Available function definitions
            tool_choice: Tool selection strategy ("auto", "required", "none")
            return_metadata: If True, returns tuple of (response, CompressionMetadata)
            **kwargs: Additional parameters (max_tokens, etc.)

        Returns:
            If return_metadata=False: ChatCompletion response from OpenAI API
            If return_metadata=True: Tuple of (ChatCompletion, CompressionMetadata)

        Raises:
            Exception: Any errors from OpenAI API (logged to wandb)
        """
        start_time = time.time()

        logger.debug(
            f"🔄 Processing {len(input_messages)} messages with {self.active_memory_key}"
        )

        model_def = self.get_model_config()
        input_token_count = get_token_count(
            input_messages, model_name=model_def.litellm_name
        )

        # Apply memory processing
        compressed_view, compressed_token_count = self.memory_processor.apply_strategy(
            input_messages,
            self.active_memory_key,
            input_token_count=input_token_count,
            llm_client=self,
        )

        # Calculate compression metrics
        if compressed_token_count is None:
            compressed_token_count = get_token_count(
                compressed_view, model_name=model_def.litellm_name
            )

        compression_ratio = (
            compressed_token_count / input_token_count if input_token_count > 0 else 1.0
        )

        # Record compressed trace entry for this call
        self._trace_step_counter += 1
        trace_entry = CompressedTraceEntry(
            step=self._trace_step_counter,
            input_token_count=input_token_count,
            compressed_token_count=compressed_token_count,
            compression_ratio=compression_ratio,
            memory_method=self.active_memory_key,
            compressed_messages=compressed_view,
        )
        self._compressed_trace_buffer.append(trace_entry)

        # Sanitize kwargs (remove model if passed by benchmark)
        kwargs.pop("model", None)

        try:
            # Build request parameters
            request_params = {
                "model": model_def.litellm_name,
                "messages": compressed_view,
                "api_base": model_def.api_base,
                "api_key": model_def.api_key,
                "drop_params": True,
            }

            # Merge config kwargs
            request_params.update(self.get_model_kwargs_from_config())

            # Merge runtime kwargs (overrides config)
            request_params.update(kwargs)

            # Only add tools and tool_choice if provided
            if tools is not None:
                request_params["tools"] = tools
            if tool_choice is not None:
                request_params["tool_choice"] = tool_choice

            response = litellm.completion(**request_params)

            if return_metadata:
                processing_time_ms = (time.time() - start_time) * 1000
                metadata = CompressionMetadata(
                    input_token_count=input_token_count,
                    compressed_token_count=compressed_token_count,
                    compression_ratio=compression_ratio,
                    memory_method=self.active_memory_key,
                    processing_time_ms=processing_time_ms,
                    loop_detected=False,
                    strategy_metadata={},
                    compressed_messages=compressed_view
                    if kwargs.get("include_messages") else None,
                    original_messages=input_messages
                    if kwargs.get("include_messages") else None,
                )
                return response, metadata

            return response

        except Exception as e:
            logger.error(f"💥 Generation Failed: {str(e)}")
            raise e

    @weave.op()
    def generate_plain(
        self,
        input_messages: Iterable[ChatCompletionMessageParam],
        **kwargs,
    ) -> Union[ChatCompletion, Any]:
        """
        Execute LLM request for evaluation. No memory processing applied. Model defaults to GPT-4.1
        Exception: Any errors from OpenAI API (logged to wandb)
        """
        kwargs.pop("model", None)
        try:
            # Try to find in registry
            model_def = self.get_model_config()

            request_params = {
                "model": model_def.litellm_name,
                "messages": input_messages,
                "api_base": model_def.api_base,
                "api_key": model_def.api_key,
                "drop_params": True,
                **kwargs,
            }

            response = litellm.completion(**request_params)
            return response

        except Exception as e:
            logger.error(f"💥 Generation Failed: {str(e)}")
            raise e
