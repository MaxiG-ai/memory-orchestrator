# memory-orchestrator

Memory orchestration layer for LLM tool-use experiments.

This repository wraps model calls with interchangeable memory strategies so you can benchmark how different context-management techniques affect agentic tool use. It was built for a master thesis and currently focuses on three documented strategies:

- Truncation
- Progressive Summarization
- Memory Bank

## What this repository does

At runtime, the orchestrator receives an evolving conversation trace, applies one configured memory strategy, and forwards the transformed message list to the target LLM. The same orchestration layer also tracks compressed traces so runs can be inspected and compared later.

High-level flow:

1. Load experiment and model configuration.
2. Select an active model and memory strategy.
3. Count tokens for the current message trace.
4. Apply the selected memory strategy.
5. Send the transformed context to LiteLLM.
6. Record compression metadata and the compressed trace view.

## Repository map

- `src/memorch/llm_orchestrator.py` — main entry point for model calls, context switching, and compression metadata
- `src/memorch/memory_processing.py` — strategy dispatch, threshold handling, and state reset
- `src/memorch/strategies/` — concrete implementations of each memory strategy
- `src/memorch/utils/` — shared helpers for config, prompts, token counting, trace splitting, and logging
- `tests/unit/` — unit tests covering orchestration, config loading, and strategy behavior
- `docs/architecture.md` — end-to-end system overview
- `docs/strategies/` — one document per memory strategy

## Documentation index

Architecture and repository-level documentation:

- `docs/architecture.md` — how the full system fits together

Per-strategy documentation:

- `docs/strategies/truncation.md`
- `docs/strategies/progressive_summarization.md`
- `docs/strategies/memory_bank.md`

Configuration examples:

- `docs/config.example.toml`
- `docs/model_config.example.toml`

## Core architecture

```text
User/benchmark trace
        |
        v
LLMOrchestrator.generate_with_memory_applied()
        |
        v
MemoryProcessor.apply_strategy()
        |
        +--> truncation
        +--> progressive_summarization
        +--> memory_bank
        +--> no_strategy
        |
        v
Compressed/transformed message list
        |
        v
litellm.completion(...)
        |
        v
Response + compression metadata + compressed trace history
```

Two strategies (`truncation` and `progressive_summarization`) are threshold-sensitive and only activate when the configured token threshold is exceeded. `memory_bank` runs every step because it maintains its own internal state.

## Setup

This project uses `uv`.

1. Clone the repository.
2. Copy the example config files:
   - `docs/config.example.toml` -> `config.toml`
   - `docs/model_config.example.toml` -> `model_config.toml`
3. Adjust model registry entries and experiment settings.
4. Install dependencies and run commands via `uv run ...`.

Example:

```bash
uv run pytest
```

## Configuration model

The runtime configuration is assembled from two TOML files:

- `config.toml` defines the experiment, enabled models, enabled memory methods, thresholds, and per-strategy settings.
- `model_config.toml` defines the model registry entries consumed by LiteLLM.

`memorch.utils.config.load_configs()` merges both into a single `ExperimentConfig` object that is then passed into `LLMOrchestrator`.

## Strategy overview

### Truncation

Keeps the original task-defining user message and only the most recent tool interaction(s). This is the simplest compression strategy and acts as a strong baseline.

### Progressive Summarization

When the trace crosses a threshold, summarizes historical context with an auxiliary LLM and reconstructs the context as a system summary plus the user query.

### Memory Bank

Builds a dual-store memory system over prior tool outputs. Raw outputs are kept in a fact store, while an observer model summarizes them into an embedding-backed insight store for retrieval.

## Tests

Run the unit suite with:

```bash
uv run pytest
```

If you only want to inspect documentation-related consistency while editing docs, a fast smoke check is:

```bash
uv run pytest tests/unit/test_config.py tests/unit/test_progressive_summarization.py tests/unit/test_truncation_strategy.py tests/unit/test_memory_bank_strategy.py
```

## Current improvement opportunities

Based on the current repository structure, the biggest opportunities are:

- add a small CLI or runnable example to make the framework easier to try manually
- add a diagram or trace example showing how message lists change step by step
- tighten configuration validation so strategy-specific required fields are enforced more explicitly
- add integration tests that exercise the full orchestrator-to-strategy pipeline with mocked model calls
- unify the strategy docs and code examples so configuration defaults always stay synchronized

## Migration note

- The webserver was removed because it remained unused for the thesis.
