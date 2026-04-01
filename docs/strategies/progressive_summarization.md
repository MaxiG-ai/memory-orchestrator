# Progressive Summarization Strategy

This document explains progressive summarization from both a conceptual and a technical perspective.

## High-level overview

Progressive summarization compresses long histories by asking an auxiliary LLM to rewrite prior conversation state into a summary. Instead of keeping the full raw trace, it produces a smaller prompt made of:

- a system summary message generated from prior history
- the first user query as the task anchor

This strategy trades exact raw history for a denser natural-language representation.

## When it runs

Progressive summarization is threshold-sensitive.

`MemoryProcessor.apply_strategy()` only invokes it when the current input token count meets or exceeds the active compact threshold. Below threshold, the original trace is left unchanged.

Relevant files:

- `src/memorch/memory_processing.py`
- `src/memorch/strategies/progressive_summarization/prog_sum.py`
- `src/memorch/strategies/progressive_summarization/prog_sum.prompt.md`

## High-level flow

```text
full message trace
   |
   v
extract first user message text
   |
   v
render summarization prompt
   |
   v
call auxiliary summarizer model
   |
   v
extract summary text
   |
   v
build final messages:
   [system(summary), user(anchor)]
```

## Why this strategy exists

This strategy tests whether an LLM-generated summary can preserve enough task-relevant information to let the main model continue accurately after context compression.

Compared with truncation, it keeps more historical information in compressed form. Compared with Memory Bank, it does not maintain a structured external retrieval store.

## Technical implementation

Main function:

```python
summarize_conv_history(messages, llm_client, summary_prompt_path, summarizer_model)
```

The implementation currently:

1. Requires an `llm_client` with `generate_plain()`.
2. Extracts the first user message text using `get_first_user_text(messages)`.
3. Builds a `PromptManager` for `prog_sum.prompt.md`.
4. Renders a system prompt that includes the user query.
5. Sends the full message list to the summarizer model as serialized text.
6. Extracts summary text from the response.
7. Returns a message list with the summary as a system message followed by the user anchor.

## Prompt construction details

The summarizer call uses two messages:

- system: the rendered summarization instructions
- user: a literal string containing `Conversation history to compress:\n{messages}`

That means the historical trace is currently passed as Python-style serialized message data, not as a specially structured JSON schema.

## Output prompt shape

Resulting prompt shape:

```text
system(summary of prior history)
user(first user query)
```

The summary is placed first because the code comments explicitly note OpenAI-style ordering expectations for system messages.

## Configuration

Example config:

```toml
[memory_strategies.progressive_summarization]
type = "progressive_summarization"
summarizer_model = "gpt-4-1-mini"
summary_prompt = "memorch/strategies/progressive_summarization/prog_sum.prompt.md"
```

Relevant config fields from `MemoryDef`:

- `summary_prompt`
- `summarizer_model`

## Dependencies

Internal dependencies:

- `memorch.utils.llm_helpers.extract_content`
- `memorch.utils.prompt_manager.PromptManager`
- `memorch.utils.split_trace.get_first_user_text`
- `memorch.utils.logger.get_logger`

Runtime dependency:

- an auxiliary LLM reachable through `llm_client.generate_plain()`

## Important behavioral assumptions

The current implementation assumes:

- the first user message is the best long-term task anchor
- a natural-language summary is sufficient to replace full history
- the summarizer returns non-empty text
- serializing the full message list into one string is acceptable input format for summarization

There is also an in-code TODO noting that the implementation should not depend so heavily on message ordering.

## Strengths

- compresses much more history than truncation
- relatively simple architecture compared with Memory Bank
- configurable summarizer prompt and summarizer model
- easy to benchmark because the output prompt shape is compact and stable

## Weaknesses

- summary quality depends entirely on the auxiliary LLM
- raw history is not recoverable after summarization
- can omit details or introduce abstraction errors
- currently relies on message ordering assumptions and the first-user anchor heuristic

## Related tests

Primary test file:

- `tests/unit/test_progressive_summarization.py`

Those tests are the best reference for expected behavior, failure modes, and prompt construction assumptions.

## Improvement opportunities

Recommended improvements for this strategy:

1. Add before/after message examples in the docs and tests.
2. Replace plain stringification of `messages` with a more controlled serialization format.
3. Tighten the prompt contract so the summary output structure is more predictable.
4. Consider preserving the latest user message as well when traces become multi-turn.
5. Add config validation that ensures `summarizer_model` is present for this strategy when needed.
