import os
import tomllib as tomli
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

from .logger import get_logger

logger = get_logger("Config")


class ModelDef(BaseModel):
    model_config = {"extra": "allow"}

    litellm_name: str
    context_window: int
    provider: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None


class MemoryDef(BaseModel):
    type: str
    target_summary_length: Optional[int] = None
    auto_compact_threshold: Optional[int] = None

    # Memory Bank strategy fields
    embedding_model: Optional[str] = "BAAI/bge-large-en-v1.5"
    top_k: Optional[int] = 3
    observer_model: Optional[str] = "gpt-4-1-mini"
    max_chars_per_record: Optional[int] = 2000

    # Fields for Progressive Summarization
    summary_prompt: str = "memorch/strategies/progressive_summarization/prog_sum.prompt.md"
    summarizer_model: Optional[str] = None

    # ACE strategy fields
    generator_model: Optional[str] = "gpt-4-1-mini"
    reflector_model: Optional[str] = "gpt-4-1-mini"
    curator_model: Optional[str] = "gpt-4-1-mini"
    curator_frequency: Optional[int] = 1
    playbook_token_budget: Optional[int] = 4096


class ExperimentConfig(BaseModel):
    experiment_name: str
    results_dir: str
    log_dir: str
    logging_level: str
    input_file: str
    proc_num: int = 1
    benchmark_sample_size: Optional[int] = None
    selected_test_cases: Optional[List[str]] = None
    enabled_models: List[str]
    enabled_memory_methods: List[str]
    compact_thresholds: List[int]

    # Haystack experiment settings
    max_messages_after_compression: Optional[int] = None  # None disables the check
    haystack_thresholds: Optional[List[int]] = None  # token count targets for haystack

    # Maps strategy name -> config
    memory_strategies: Dict[str, MemoryDef]

    # Maps model key -> config (Populated from model_config.toml)
    model_registry: Dict[str, ModelDef] = Field(default_factory=dict)


def load_configs(
    exp_path="config.toml", model_path: Optional[str] = None
) -> ExperimentConfig:
    """
    Loads and merges the experiment config with an optional model registry.
    Also sets the global logging level based on the config.

    Model definitions can be supplied in three ways (in order of preference):
    1. A separate ``model_path`` TOML file (e.g. ``model_config.toml`` in the
       execution repository) whose ``[models]`` table is merged into the config.
    2. A ``[models]`` table embedded directly in ``exp_path``.
    3. Omitted entirely when a pre-built :class:`ExperimentConfig` is injected
       via ``LLMOrchestrator(config=...)``.

    Args:
        exp_path: Path to the experiment config TOML file. Must exist.
        model_path: Optional path to a separate model-registry TOML file.
            When ``None`` (default) no separate file is loaded; model
            definitions may be embedded in ``exp_path`` or left empty.
    """
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"Missing experiment config file: {exp_path}")

    with open(exp_path, "rb") as f:
        exp_data = tomli.load(f)

    # Load separate model registry file when provided
    if model_path is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model config file: {model_path}")
        with open(model_path, "rb") as f:
            model_data = tomli.load(f)
        # Inject registry into experiment config for a single unified object
        # The 'models' key in toml becomes 'model_registry' in Pydantic
        exp_data["model_registry"] = model_data.get("models", {})
    elif "models" in exp_data:
        # Allow models to be embedded directly in the experiment config under [models]
        exp_data.setdefault("model_registry", exp_data.pop("models"))

    try:
        config = ExperimentConfig(**exp_data)

        # Set global logging level from config
        from .logger import set_global_log_level

        set_global_log_level(config.logging_level)

        return config
    except ValidationError as e:
        logger.error("❌ Configuration Error")
        logger.error(str(e))
        raise
