# Truncation Strategy

This document explains the truncation strategy at both a high level and an implementation level.

## High-level overview

Truncation is the simplest memory strategy in the repository. It reduces context size by discarding most historical messages and preserving only the parts considered most important for continuing the task:

- the first user message, treated as the task anchor
- the last tool interaction, or the last `n` tool interactions if configured in code

This makes truncation a strong baseline because it is:

- easy to reason about
- deterministic
- cheap to run
- independent of auxiliary LLM calls or embedding models

## When it runs

Truncation is threshold-sensitive.

That means `MemoryProcessor.apply_strategy()` only activates it when the incoming token count is greater than or equal to the current compact threshold. If the trace is still below threshold, the original messages are returned unchanged.

Relevant dispatch path:

- `src/memorch/memory_processing.py`
- `src/memorch/strategies/truncation/truncation.py`

## Why this strategy exists

In tool-use traces, older messages often become less important than:

- the original task definition
- the most recent tool call and response

Truncation tests the hypothesis that preserving only these anchors may be enough for the model to continue successfully, while dramatically reducing token usage.

## Prompt shape after truncation

Input shape:

```text
system?
user(task)
assistant
tool
assistant
tool
...
assistant(tool_calls)
tool(result)
```

Output shape:

```text
user(first task message)
assistant(last tool call)
tool(last tool result)
```

If `keep_last_n_tool_interactions > 1`, the output includes more than one recent interaction block.

## Technical implementation

File:

- `src/memorch/strategies/truncation/truncation.py`

Main function:

```python
truncate_messages(messages, keep_last_n_tool_interactions=1) -> List[Dict]
```

Core logic:

1. Extract user messages with `get_user_message(messages)`.
2. Keep only the first user message as the task anchor.
3. Repeatedly extract the last tool interaction with `get_last_tool_interaction(old_messages)`.
4. Append those tool interaction messages to the return list.
5. Stop after `keep_last_n_tool_interactions` blocks.

## Detailed control flow

```text
messages
  |
  v
get_user_message(messages)
  |
  v
keep first user message only
  |
  v
loop:
  get_last_tool_interaction(old_messages)
  append interaction to result
  remove interaction from old_messages
until desired number of interactions is kept
```

## Code-level behavior

Important details from the implementation:

- It intentionally keeps only `user_msgs[:1]`.
- If more than one user message exists, it logs a warning.
- It works by walking backward through the trace, one tool interaction at a time.
- It does not summarize or reinterpret anything; it only slices the trace.

This means the strategy is loss-heavy but semantically transparent: the kept messages are exact originals.

## Dependencies

Internal dependencies:

- `memorch.utils.split_trace.get_user_message`
- `memorch.utils.split_trace.get_last_tool_interaction`
- `memorch.utils.logger.get_logger`

There are no external model dependencies.

## Configuration

Config section:

```toml
[memory_strategies.truncation]
type = "truncation"
```

There are currently no TOML-exposed strategy-specific parameters for truncation. The `keep_last_n_tool_interactions` parameter is code-level only.

## Strengths

- cheapest strategy in the repository
- deterministic and easy to debug
- no extra latency from auxiliary model calls
- preserves exact raw messages for the most recent interaction

## Weaknesses

- aggressively discards intermediate reasoning and history
- assumes the first user message is always the correct task anchor
- may lose critical context if relevant information lives outside the last interaction block
- strategy configurability is minimal from TOML today

## Related tests

Primary test file:

- `tests/unit/test_truncation_strategy.py`

Those tests are the best place to confirm edge-case expectations around message extraction and retained interactions.

## Improvement opportunities

Recommended improvements for this strategy:

1. Expose `keep_last_n_tool_interactions` in config.
2. Add a worked example in tests or docs showing exactly how `get_last_tool_interaction()` partitions a realistic trace.
3. Clarify whether the first user message or the latest user message should be the anchor in multi-user-turn traces.
4. Add docs for behavior when no tool interaction exists yet.
