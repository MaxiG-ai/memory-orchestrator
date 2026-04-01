# Memory Bank Strategy

This document explains the Memory Bank strategy at both the system and implementation levels.

## High-level overview

Memory Bank is a stateful retrieval-based memory strategy for tool-use traces.

Its core idea is to separate:

- precision: preserve exact raw tool outputs
- reasoning: index natural-language summaries of those tool outputs for semantic retrieval

To do that, Memory Bank uses a dual-store architecture:

- `FactStore` keeps raw `InteractionRecord` objects keyed by `trace_id`
- `InsightStore` keeps summary embeddings keyed by the same `trace_id`

At each step, the strategy can ingest new tool outputs, retrieve the most relevant prior interactions, and rebuild a compact prompt that contains only:

- retrieved context
- the user anchor
- the latest tool interaction

## When it runs

Memory Bank is not threshold-gated.

`MemoryProcessor.apply_strategy()` runs it on every step because it needs to maintain and consult state continuously.

Relevant files:

- `src/memorch/memory_processing.py`
- `src/memorch/strategies/memory_bank/memory_bank_strategy.py`
- `src/memorch/strategies/memory_bank/models.py`
- `src/memorch/strategies/memory_bank/fact_store.py`
- `src/memorch/strategies/memory_bank/insight_store.py`
- `src/memorch/strategies/memory_bank/observer.py`
- `src/memorch/strategies/memory_bank/ingestion.py`
- `src/memorch/strategies/memory_bank/retrieval.py`
- `src/memorch/strategies/memory_bank/observer.prompt.md`

## Architecture at a glance

```text
                     +----------------------+
                     |   Incoming messages  |
                     +----------+-----------+
                                |
                                v
                   +------------+-------------+
                   | apply_memory_bank_strategy|
                   +------+--------------------+
                          |
        +-----------------+------------------+
        |                                    |
        v                                    v
  ingest new tool outputs             retrieve prior memory
        |                                    |
        v                                    v
+---------------+                    +---------------+
|   FactStore   |<---- trace_id ---->|  InsightStore |
| raw records   |                    | summaries +   |
| by trace_id   |                    | embeddings    |
+---------------+                    +---------------+
        |                                    |
        +-----------------+------------------+
                          |
                          v
                reconstructed prompt context
```

## Core data model

The central record type is `InteractionRecord`.

It captures one tool interaction with fields such as:

- `trace_id`
- `step_id`
- `tool_name`
- `raw_input`
- `raw_output`
- `timestamp`

`trace_id` is the join key between both stores.

## Runtime state

Memory Bank persists a `MemoryBankState` object across task steps.

Important fields:

- `fact_store`
- `insight_store`
- `step_count`
- `_embedding_model`
- `call_log`

Lifecycle behavior:

- initialized lazily on first use
- persists across steps within one task
- reset between tasks
- keeps the embedding model loaded across resets for reuse

## End-to-end flow

### 1. Step starts

`apply_memory_bank_strategy()` increments `state.step_count` and ensures the embedding model and `InsightStore` are initialized.

### 2. User query anchor is extracted

The strategy uses `get_first_user_text(messages)` to recover the task anchor used for observer summaries and retrieval query generation.

### 3. Tool outputs are ingested

There are two ingestion modes.

#### Cold-start backfill mode

If the insight store is empty, the strategy walks backward through the existing trace and ingests all historical tool interactions into the stores.

This lets the strategy bootstrap even if it is activated after tool activity already exists in the current message list.

#### Incremental ingestion mode

If the insight store already has entries, the strategy extracts only the new tool outputs from the current trace and adds them.

### 4. Loop detection is applied

Every ingested tool call contributes a serialized `(tool_name, raw_input)` key to `state.call_log`.

If the same key appears at or above `LOOP_THRESHOLD`, the strategy raises an error.

This is a strategy-specific safeguard against repeated identical tool calls.

### 5. Relevant prior context is retrieved

The strategy builds a retrieval query from the current user task:

```text
Retrieve relevant information to answer the task: {user_query}
```

It then:

1. searches the `InsightStore` semantically
2. gets top-k `trace_id`s
3. fetches the corresponding raw records from the `FactStore`
4. formats those into a system message

### 6. New prompt is constructed

The final processed message list is:

```text
system(retrieved context)
user(anchor messages)
last tool interaction messages
```

If no memory has been ingested yet, the strategy passes the original messages through unchanged.

## Subcomponents

### FactStore

File: `fact_store.py`

An in-memory dictionary mapping `trace_id -> InteractionRecord`.

Purpose:

- preserve raw tool outputs as ground truth
- support exact lookup after semantic retrieval

### InsightStore

File: `insight_store.py`

An in-memory vector store over summary embeddings.

Purpose:

- embed observer summaries
- perform semantic search over prior interactions
- return matching `trace_id`s

It currently stores entries as a Python list of dictionaries and computes similarities with NumPy.

### Observer

File: `observer.py`

The Observer is an auxiliary LLM component that summarizes what a tool execution achieved.

Its summary is intentionally shorter and more retrieval-friendly than the full tool output. It highlights key identifiers and results while avoiding full JSON duplication.

### Ingestion pipeline

File: `ingestion.py`

For each extracted tool output tuple:

1. create an `InteractionRecord`
2. store it in `FactStore`
3. summarize it through the Observer
4. add the summary to `InsightStore`

### Retrieval pipeline

File: `retrieval.py`

Retrieval takes a query, looks up matching summaries, fetches raw records, truncates raw output strings if needed, and formats a system message for prompt injection.

## Configuration

Example:

```toml
[memory_strategies.memory_bank]
type = "memory_bank"
observer_model = "gpt-4-1-mini"
embedding_model = "BAAI/bge-large-en-v1.5"
top_k = 3
max_chars_per_record = 2000
```

Relevant config fields:

- `observer_model`
- `embedding_model`
- `top_k`
- `max_chars_per_record`

## Dependencies

Internal dependencies:

- `split_trace` helpers for finding user messages, tool interactions, and tool outputs
- `token_count` for reporting compressed token counts
- `llm_helpers.extract_content`
- `logger`

External dependencies:

- `FlagEmbedding.FlagModel`
- `numpy`
- an LLM backend for observer summarization

## Strengths

- separates retrieval semantics from raw data preservation
- can recover important exact values from prior steps
- bootstraps from existing trace history when the store starts empty
- more structured than simple summarization

## Weaknesses

- more operationally complex than truncation or progressive summarization
- depends on both embeddings and an observer model
- in-memory stores are ephemeral and task-local only
- retrieval quality depends on observer summary quality and embedding behavior

## Related tests

Primary test file:

- `tests/unit/test_memory_bank_strategy.py`

This is one of the most important test files in the repository because it covers the dual-store data model, observer behavior, retrieval formatting, backfill logic, and end-to-end strategy orchestration.

## Improvement opportunities

Recommended improvements for Memory Bank:

1. Add a worked example showing one tool output being ingested and later retrieved.
2. Clarify whether retrieval should include deduplication or recency weighting in the future.
3. Add stronger serialization guarantees for raw tool outputs before observer calls.
4. Consider extracting retrieval ranking policy into a dedicated configurable component.
5. Document performance expectations for larger numbers of stored entries since the current implementation is in-memory and linear in several places.
