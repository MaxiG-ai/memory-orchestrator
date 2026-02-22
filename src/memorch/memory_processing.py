import weave
from typing import Any, Dict, List, Optional, Tuple

from memorch.utils.logger import get_logger
from memorch.utils.token_count import get_token_count
from memorch.utils.trace_processing import detect_tail_loop
from memorch.utils.config import ExperimentConfig

from memorch.strategies.progressive_summarization.prog_sum import summarize_conv_history
from memorch.strategies.truncation.truncation import truncate_messages
from memorch.strategies.ace.ace_strategy import ACEState, apply_ace_strategy
from memorch.strategies.memory_bank import MemoryBankState, apply_memory_bank_strategy

logger = get_logger("MemoryProcessor")


class MemoryProcessor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.current_summary: str = ""
        self._ace_state = ACEState()
        self._memory_bank_state: Optional[MemoryBankState] = None

    def reset_state(self):
        """Called by Orchestrator to reset memory between runs."""
        self.current_summary = ""
        self._ace_state.reset()
        if self._memory_bank_state is not None:
            self._memory_bank_state.reset()
        logger.info("🧠 Memory State Reset")

    @weave.op(enable_code_capture=False)
    def apply_strategy(
        self,
        messages: List[Dict],
        memory_key: str,
        input_token_count: int,
        llm_client: LLMOrchestrator,
    ) -> Tuple[List[Dict], Optional[int]]:
        """
        Apply the configured memory strategy to the incoming messages.
        """
        settings = self.config.memory_strategies[memory_key]
        logger.debug(f"🧠 Applying Memory Strategy: {settings.type}")

        # Loop detection to prevent infinite context growth
        # Pure rate limit is in llm_orchestrator, this should prevent loops before hard limit
        if len(messages) > 20 and detect_tail_loop(
            messages, threshold=4, max_pattern_len=5
        ):
            logger.error(
                f"🚨 Infinite loop detected in last {len(messages)} messages. Aborting."
            )
            return [
                {"role": "system", "content": "Infinite loop detected; aborting."}
            ], None

        ### MEMORY STRATEGY APPLICATION LOGIC ###
        #########################################
        
        ## STRATEGIES APPLIED REGARDLESS OF TOKEN COUNT ##
        if settings.type == "ace":
            processed_messages, output_token_count = self._apply_ace(
                messages=messages,
                token_count=input_token_count,
                settings=settings,
                llm_client=llm_client,
            )
            return processed_messages, output_token_count

        if settings.type == "memory_bank":
            processed_messages, output_token_count = self._apply_memory_bank(
                messages=messages,
                token_count=input_token_count,
                settings=settings,
                llm_client=llm_client,
            )
            return processed_messages, output_token_count
        

        ## STRATEGIES APPLIED BASED ON TOKEN COUNT ##
        if input_token_count < self.config.compact_threshold:
            return messages, input_token_count
        else:
            logger.debug(
                f"🧠 Pre-Processing Token Count: {input_token_count}, exceeds compact_threshold={self.config.compact_threshold}"
            )
            # Truncation
            if settings.type == "truncation":
                processed_messages, output_token_count = self._apply_truncation(
                    messages=messages, 
                    token_count=input_token_count
                )
            # Progressive Summarization
            elif settings.type == "progressive_summarization":
                processed_messages, output_token_count = (
                    self._apply_progressive_summarization(
                        messages=messages,
                        token_count=input_token_count,
                        settings=settings,
                        llm_client=llm_client,
                    )
                )
            else:
                logger.warning(
                    f"🧠 Unknown memory strategy type: {settings.type}. Applying no_strategy; returning original messages."
                )
                output_token_count = input_token_count
                return messages, output_token_count

        return processed_messages, output_token_count

    @weave.op(enable_code_capture=False)
    def _apply_truncation(
        self, 
        messages: List[Dict], 
        token_count: int
    ) -> Tuple[List[Dict], int]:
        """Applies Truncation to a message trace by cutting conversation history, 
        keeping only the last user query and tool interaction."""
        logger.debug(
            f"🧠 Applying Truncation Strategy. Current query with {token_count} tokens"
        )

        truncated_conv = truncate_messages(messages)
        return truncated_conv, get_token_count(truncated_conv)

    @weave.op(enable_code_capture=False)
    def _apply_progressive_summarization(
        self,
        messages: List[Dict],
        token_count: int,
        settings,
        llm_client: Optional[Any],
    ) -> Tuple[List[Dict], int]:
        """Summarizes archived context when token threshold is exceeded.

        Summarizes all messages before user query.
        """
        logger.debug(
            f"🧠 Applying Progressive Summarization. Current query with {token_count} tokens"
        )
        summarized_conv = summarize_conv_history(
            messages=messages,
            llm_client=llm_client,
            summarizer_model=settings.summarizer_model,
            summary_prompt_path=settings.summary_prompt,
        )
        return summarized_conv, get_token_count(summarized_conv)

    @weave.op(enable_code_capture=False)
    def _apply_ace(
        self,
        messages: List[Dict],
        token_count: int,
        settings,
        llm_client: Optional[Any],
    ) -> Tuple[List[Dict], int]:
        """Applies ACE strategy by delegating to ace_strategy module."""
        logger.debug(
            f"🧠 Applying ACE Strategy. Current query with {token_count} tokens"
        )
        processed = apply_ace_strategy(
            messages, 
            llm_client, 
            settings, self._ace_state
        )
        return processed, get_token_count(processed)

    @weave.op(enable_code_capture=False)
    def _apply_memory_bank(
        self,
        messages: List[Dict],
        token_count: int,
        settings,
        llm_client: Optional[Any],
    ) -> Tuple[List[Dict], int]:
        """Applies memory bank strategy - vector-based retrieval over past interactions.

        Lazily initializes state on first call to avoid loading embedding model unnecessarily.
        """
        logger.debug(
            f"🧠 Applying Memory Bank Strategy. Current query with {token_count} tokens"
        )

        # Lazy initialization of memory bank state (model loaded in apply_memory_bank_strategy)
        if self._memory_bank_state is None:
            self._memory_bank_state = MemoryBankState()

        processed, _ = apply_memory_bank_strategy(
            messages, llm_client, settings, self._memory_bank_state
        )
        return processed, get_token_count(processed)
