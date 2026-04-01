# ACE Strategy

This document explains the ACE strategy from both a high-level and a technical viewpoint.

## High-level overview

ACE stands for Agentic Context Engineering.

In this repository, ACE is a stateful memory strategy that tries to improve future steps in a task by maintaining a structured playbook. Instead of compressing context only by deleting or summarizing prior messages, ACE learns guidance over time and reinjects that guidance into the prompt.

Its runtime cycle has three conceptual subcomponents:

- Generator: produces a reasoning trace for the next step using the current playbook
- Reflector: inspects the previous step and evaluates how the playbook was used
- Curator: edits the playbook based on reflections and usage statistics

The result is a self-updating prompt scaffold that evolves during a task.

## When it runs

ACE is not threshold-gated.

`MemoryProcessor.apply_strategy()` runs ACE every step, regardless of token count, because ACE maintains task-local state that would not make sense to activate only after a threshold.

Relevant files:

- `src/memorch/memory_processing.py`
- `src/memorch/strategies/ace/ace_strategy.py`
- `src/memorch/strategies/ace/generator.py`
- `src/memorch/strategies/ace/reflector.py`
- `src/memorch/strategies/ace/curator.py`
- `src/memorch/strategies/ace/playbook_utils.py`
- `src/memorch/strategies/ace/prompts/*.prompt.md`

## High-level ACE loop

```text
step N input messages
   |
   v
load current ACEState
   |
   +--> Reflect on previous step if a reasoning trace exists
   |
   +--> Update playbook bullet counts
   |
   +--> Curate playbook on configured frequency
   |
   +--> Generate next-step reasoning trace from playbook
   |
   v
return:
  [system(playbook)] + original messages + [system(reasoning trace)]
```

## Core runtime state

ACE persists a task-scoped `ACEState` object.

Fields:

- `playbook`
- `next_global_id`
- `last_reflection`
- `last_bullet_ids`
- `last_reasoning_trace`
- `last_predicted_answer`
- `step_count`

This state is initialized in `MemoryProcessor` and reset between tasks.

## Technical orchestration

The main orchestration function is:

```python
apply_ace_strategy(messages, llm_client, settings, state)
```

Its execution order is:

1. Increment `state.step_count`.
2. Read `curator_frequency` from config.
3. Extract the first user message as the task description.
4. If a prior reasoning trace exists, run the Reflector.
5. Use reflector output to update playbook bullet counts.
6. If the current step matches curation frequency, run the Curator.
7. Run the Generator to prepare the next-step reasoning trace.
8. Store the new reasoning trace and used bullet IDs in state.
9. Inject both the playbook and the reasoning trace into the returned message list.

## Prompt shape after ACE processing

Returned prompt shape:

```text
system(ACE playbook)
...original messages...
system(ACE reasoning trace for current step)
```

This means ACE augments the original trace rather than replacing it outright.

## Reflector stage

Reflector runs only when `state.last_reasoning_trace` exists.

Inputs include:

- the task
- the previous reasoning trace
- the previous predicted answer
- a placeholder environment feedback string
- the playbook bullets referenced by the previous step

Outputs include:

- `reflection_text`
- `bullet_tags`

Those bullet tags are then used to update bullet counts in the playbook.

## Curator stage

Curator runs when:

```text
state.step_count % curator_frequency == 0
```

Inputs include:

- current playbook
- latest reflection
- placeholder additional context
- current step
- playbook token budget
- playbook statistics
- next global bullet id

Outputs include:

- updated playbook
- updated next-global-id
- curation operations

The curator is responsible for keeping the playbook useful and within budget.

## Generator stage

Generator always runs to create a reasoning trace for the next step.

Inputs include:

- task
- playbook
- a short context string derived from the last three messages
- latest reflection

Outputs include:

- `reasoning_trace`
- `bullet_ids_used`

Those outputs are stored in `ACEState` for the next reflection cycle.

## Configuration

Example config section:

```toml
[memory_strategies.ace]
type = "ace"
generator_model = "gpt-4-1-mini"
reflector_model = "gpt-4-1-mini"
curator_model = "gpt-4-1-mini"
curator_frequency = 1
playbook_token_budget = 4096
```

Relevant `MemoryDef` fields:

- `generator_model`
- `reflector_model`
- `curator_model`
- `curator_frequency`
- `playbook_token_budget`

## Dependencies

Internal dependencies:

- `playbook_utils` for playbook templates, bullet extraction, counts, and stats
- `generator.py`
- `reflector.py`
- `curator.py`
- `memorch.utils.split_trace.get_first_user_text`
- `memorch.utils.logger.get_logger`

Runtime dependencies:

- auxiliary LLM access through `llm_client`
- prompt files under `src/memorch/strategies/ace/prompts/`

## Strengths

- explicitly models learning across steps within a task
- keeps a reusable playbook rather than only compressing raw trace text
- modular architecture with clear subcomponents
- configurable frequency and budget controls

## Weaknesses

- more complex than the other strategies
- adds multiple auxiliary-model dependencies
- currently uses placeholder environment feedback and placeholder additional context
- prompt and playbook evolution can be harder to debug than direct trace compression

## Related tests

Primary test file:

- `tests/unit/test_ace_strategy.py`

That test suite is important because ACE behavior depends on multi-step state transitions that are easier to understand through mocked examples than through static code inspection alone.

## Improvement opportunities

Recommended improvements for ACE:

1. Add a dedicated architecture diagram showing Generator, Reflector, and Curator interactions.
2. Document the playbook text format more explicitly, including bullet IDs and count semantics.
3. Replace placeholder environment feedback and additional context with real task signals when available.
4. Add docs with a worked multi-step example showing how `ACEState` evolves over three turns.
5. Add stronger guarantees or schemas for generator and reflector outputs if prompt drift becomes a problem.
