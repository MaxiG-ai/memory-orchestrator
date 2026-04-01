# Repository Architecture Overview

This document explains how the repository is organized, how a request flows through the system, and how the individual memory strategies fit into the overall design.

## 1. Purpose

`memory-orchestrator` is an experiment framework for evaluating memory strategies in LLM tool-use workloads. Its main job is to sit between an agent/benchmark and an LLM backend, rewrite the conversation context according to a chosen memory strategy, and record what compression happened.

The architecture separates:

- orchestration of model calls
- memory-strategy dispatch
- strategy-specific state and algorithms
- shared utilities such as config, token counting, prompts, and trace parsing

## 2. Top-level module map

```text
src/memorch/
├── llm_orchestrator.py          # Runtime entry point for model calls
├── memory_processing.py         # Strategy selection and application
├── exceptions.py                # Shared custom exceptions
├── strategies/
│   ├── truncation/
│   ├── progressive_summarization/
│   ├── memory_bank/
│   └── no_strategy.py
└── utils/
    ├── config.py
    ├── token_count.py
    ├── split_trace.py
    ├── trace_processing.py
    ├── trace_history.py
    ├── llm_helpers.py
    ├── prompt_manager.py
    └── logger.py
```

Supporting files:

```text
docs/                        # Human documentation and config examples
tests/unit/                  # Unit tests covering orchestration and strategies
pyproject.toml               # Package metadata and dependencies
AGENTS.md                    # Contributor conventions for this repo
```

## 3. End-to-end request lifecycle

A single request follows this path:

```text
Input messages
   |
   v
LLMOrchestrator.generate_with_memory_applied()
   |
   +--> get active model config
   +--> count input tokens
   +--> call MemoryProcessor.apply_strategy(...)
              |
              +--> select strategy from config
              +--> run loop detection
              +--> apply chosen strategy
   |
   +--> compute compression metrics
   +--> append compressed trace entry
   +--> send request to litellm.completion(...)
   |
   v
LLM response
```

## 4. Primary runtime components

### 4.1 `LLMOrchestrator`

File: `src/memorch/llm_orchestrator.py`

This is the main runtime interface. It owns the experiment configuration, the active model/memory selection, and the trace buffer that stores compressed views.

Main responsibilities:

- load or accept an `ExperimentConfig`
- keep track of active model and memory strategy
- route each request through memory processing
- call LiteLLM with the transformed message list
- compute and expose compression metadata
- reset per-session state between tasks

Important state:

- `active_model_key`
- `active_memory_key`
- `active_compact_threshold`
- `memory_processor`
- `_compressed_trace_buffer`
- `last_compressed_view`

### 4.2 `MemoryProcessor`

File: `src/memorch/memory_processing.py`

This class is the dispatch layer between the orchestrator and concrete strategies.

Responsibilities:

- look up the selected strategy in config
- apply generic loop detection before strategy execution
- decide whether threshold-sensitive strategies should run
- keep persistent strategy state for Memory Bank
- provide a unified `(processed_messages, token_count)` output contract

The processor contains explicit strategy branches for:

- `truncation`
- `progressive_summarization`
- `memory_bank`
- `no_strategy`

## 5. Strategy taxonomy

There are two architectural categories of strategies.

### 5.1 Stateless or mostly stateless compression

These transform the visible message list directly.

- `truncation`
- `progressive_summarization`
- `no_strategy`

### 5.2 Stateful memory systems

These preserve task-level state across steps.

- `memory_bank` via `MemoryBankState`

This distinction matters because the processor must reset those states between tasks while keeping them alive across turns inside one task.

## 6. Shared utilities

### Config

File: `src/memorch/utils/config.py`

The configuration layer merges:

- experiment settings from `config.toml`
- model registry settings from `model_config.toml`

into a single `ExperimentConfig` object.

Key models:

- `ModelDef`
- `MemoryDef`
- `ExperimentConfig`

### Token counting

File: `src/memorch/utils/token_count.py`

Used by both the orchestrator and strategies to estimate the size of the current context and the compressed output.

### Trace parsing and loop detection

Files:

- `src/memorch/utils/split_trace.py`
- `src/memorch/utils/trace_processing.py`
- `src/memorch/utils/trace_history.py`

These utilities support:

- extracting user messages
- extracting last tool interactions
- parsing tool outputs from traces
- detecting repeating tail loops
- serializing compressed trace history entries

### Prompt and LLM helpers

Files:

- `src/memorch/utils/prompt_manager.py`
- `src/memorch/utils/llm_helpers.py`

These are used by LLM-backed strategies such as progressive summarization and Memory Bank observer flows.

## 7. Memory strategy integration points

```text
                             +------------------+
                             |  ExperimentConfig|
                             +---------+--------+
                                       |
                                       v
+-------------------+       +----------+----------+
| Input trace       | ----> |   MemoryProcessor   |
+-------------------+       +----------+----------+
                                       |
              +------------------------+------------------------+
              |                        |                        |
              v                        v                        v
        Truncation          Progressive Summarization        Memory Bank
```

All strategies ultimately return a transformed message list that becomes the actual prompt sent to the LLM.

## 8. Message-shape transformations

The most important architectural concept in the repository is that each strategy produces a different prompt shape.

### No strategy

```text
original messages -> original messages
```

### Truncation

```text
full history -> first user task + last tool interaction(s)
```

### Progressive summarization

```text
full history -> system summary + first user query
```

### Memory Bank

```text
messages -> retrieved-memory system message + user anchor + last tool interaction
```

## 9. State lifecycle

The repository distinguishes between per-task state and per-call state.

### Per-call state

Created and consumed within one `generate_with_memory_applied()` invocation:

- token counts
- request parameters
- compression metadata
- transformed message list

### Per-task state

Persists across multiple calls until `reset_session()`:

- Memory Bank stores and loaded embedding model
- compressed trace buffer
- trace step counter

Reset path:

```text
LLMOrchestrator.reset_session()
   -> MemoryProcessor.reset_state()
      -> MemoryBankState.reset()
   -> clear compressed trace buffer
   -> clear last compressed view
```

## 10. Configuration-driven behavior

The architecture is intentionally config-driven.

Examples:

- `enabled_models` selects which model registry entries are benchmarked
- `enabled_memory_methods` controls which strategies participate in experiments
- `compact_thresholds` defines when threshold-sensitive compression should run
- strategy sections under `memory_strategies.*` customize each implementation

This makes the repository suitable for matrix-style experimentation without changing code.

## 11. Testing architecture

The test suite is concentrated in `tests/unit/`.

Broad coverage areas include:

- config parsing and validation
- token counting and trace processing utilities
- orchestrator behavior
- strategy-specific logic for truncation, progressive summarization, and Memory Bank

The strategy docs should be read together with their unit tests because the tests often capture edge cases and implicit contracts more precisely than the current code comments do.

## 12. Architectural strengths

Current strengths of the design:

- clean separation between orchestration and strategy logic
- strategy implementations are isolated in dedicated modules
- stateful strategies explicitly model their runtime state
- config-driven experiment matrix is simple to extend
- compressed trace history creates a useful inspection surface for research work

## 13. Architectural improvement opportunities

Here are the main repository-level improvements suggested by the current implementation.

### 13.1 Add a runnable example or CLI

The architecture is clear once you read the code, but there is no obvious single entry command or minimal example for trying it manually.

### 13.2 Strengthen strategy-specific config validation

`MemoryDef` is flexible, but strategy-specific required fields are only enforced indirectly. A discriminated config model or explicit per-strategy validation would make setup failures easier to understand.

### 13.3 Standardize terminology across code and docs

The repository uses terms like memory method, memory strategy, compression, and context processing. A short glossary would improve consistency.

### 13.4 Add higher-level integration tests

Most important parts are unit-tested, but there is room for tests that simulate multi-step orchestrator flows with mocked model calls and verify state transitions end to end.

### 13.5 Add message-trace examples to docs

Because this repository transforms message arrays rather than just strings, docs would be stronger with before/after JSON examples for each strategy.

## 14. Suggested reading order

If you are new to the codebase, read in this order:

1. `README.md`
2. `docs/architecture.md`
3. `src/memorch/llm_orchestrator.py`
4. `src/memorch/memory_processing.py`
5. the strategy doc you care about under `docs/strategies/`
6. the matching strategy implementation under `src/memorch/strategies/`
7. the corresponding tests in `tests/unit/`
