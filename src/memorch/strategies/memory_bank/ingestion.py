"""
Ingestion Pipeline - Processes tool outputs into the dual-store system.

Captures tool outputs, stores raw data in FactStore, generates summaries
via Observer LLM, and stores embeddings in InsightStore.
"""

from typing import Any, Dict, List, Tuple

from memorch.strategies.memory_bank.models import InteractionRecord
from memorch.strategies.memory_bank.fact_store import FactStore
from memorch.strategies.memory_bank.insight_store import InsightStore
from memorch.strategies.memory_bank.observer import observe_tool_output
from memorch.utils.logger import get_logger

logger = get_logger("Ingestion")


def ingest_tool_outputs(
    tool_outputs: List[Tuple[str, Dict, Dict]],
    user_query: str,
    fact_store: FactStore,
    insight_store: InsightStore,
    llm_client: Any,
    observer_model: str,
    step_id: int,
) -> List[str]:
    """
    Ingest tool outputs into the dual-store system.

    For each tool output:
    1. Create InteractionRecord and store in FactStore
    2. Call Observer LLM to generate summary
    3. Embed summary and store in InsightStore

    Args:
        tool_outputs: List of (tool_name, raw_input, raw_output) tuples
        user_query: User's original task (for Observer context)
        fact_store: FactStore instance
        insight_store: InsightStore instance
        llm_client: LLM client for Observer
        observer_model: Model to use for Observer LLM
        step_id: Current step number in the task

    Returns:
        List of generated trace_ids
    """
    trace_ids = []

    for tool_name, raw_input, raw_output in tool_outputs:
        # Create and store interaction record
        record = InteractionRecord.create(
            step_id=step_id,
            tool_name=tool_name,
            raw_input=raw_input,
            raw_output=raw_output,
        )
        fact_store.store(record)

        # Generate summary via Observer LLM
        summary = observe_tool_output(
            user_query=user_query,
            tool_name=tool_name,
            raw_output=raw_output,
            llm_client=llm_client,
            model=observer_model,
        )

        # Store summary with embedding
        insight_store.add(record.trace_id, summary)

        trace_ids.append(record.trace_id)
        logger.debug(
            f"Ingested tool '{tool_name}' -> trace_id={record.trace_id[:8]}..."
        )

    return trace_ids
