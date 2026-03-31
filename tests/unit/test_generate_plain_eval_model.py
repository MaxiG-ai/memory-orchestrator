"""Tests for LLMOrchestrator.generate_plain evaluation model isolation.

generate_plain() is the method called by all evaluation/judge tasks (completeness
scoring, correctness scoring, LLM-based function call comparison).  It must always
use the configured evaluation_model rather than the active benchmarked model so that
switching the main model under test (e.g. from gpt-4-1-mini to qwen35 or glm-4-7)
does not accidentally route structured-JSON judge prompts through a model whose
output format is incompatible with the evaluation harness.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memorch.utils.config import ExperimentConfig, ModelDef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_def(litellm_name: str) -> ModelDef:
    return ModelDef(
        litellm_name=litellm_name, context_window=128_000, provider="aicore"
    )


def _make_config(
    benchmarked_model: str = "qwen35",
    evaluation_model: str | None = "gpt-4-1-mini",
) -> ExperimentConfig:
    """Build a minimal ExperimentConfig with two models in the registry."""
    return ExperimentConfig(
        experiment_name="test",
        results_dir="results",
        log_dir="logs",
        logging_level="INFO",
        input_file="data.jsonl",
        enabled_models=[benchmarked_model],
        enabled_memory_methods=["no_strategy"],
        compact_thresholds=[5000],
        memory_strategies={"no_strategy": {"type": "no_strategy"}},
        evaluation_model=evaluation_model,
        model_registry={
            benchmarked_model: _make_model_def(f"openai/{benchmarked_model}"),
            "gpt-4-1-mini": _make_model_def("sap/gpt-4.1-mini"),
            "gpt-5": _make_model_def("sap/gpt-5"),
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_plain_uses_evaluation_model_not_active_model():
    """
    generate_plain() must route requests through the evaluation_model from
    config, not through the active_model_key (the benchmarked model).

    Concretely: when the benchmarked model is 'qwen35' but evaluation_model is
    'gpt-4-1-mini', the litellm.completion call must receive the litellm_name of
    gpt-4-1-mini ('sap/gpt-4.1-mini'), not the qwen litellm_name.

    This is the core regression guard for the bug where evaluation of non-OpenAI
    models would fail because qwen/glm outputs don't match the JSON schema the
    judge prompts expect.
    """
    config = _make_config(benchmarked_model="qwen35", evaluation_model="gpt-4-1-mini")

    with (
        patch("memorch.llm_orchestrator.MemoryProcessor"),
        patch("memorch.llm_orchestrator.litellm.completion") as mock_completion,
    ):
        mock_completion.return_value = MagicMock()
        from memorch.llm_orchestrator import LLMOrchestrator

        orch = LLMOrchestrator(config=config)
        # active_model_key defaults to the first enabled model (the benchmarked model)
        assert orch.active_model_key == "qwen35"

        orch.generate_plain(input_messages=[{"role": "user", "content": "test"}])

    called_model = mock_completion.call_args.kwargs.get(
        "model",
        mock_completion.call_args.args[0] if mock_completion.call_args.args else None,
    )
    assert called_model == "sap/gpt-4.1-mini", (
        f"Expected evaluation model litellm_name 'sap/gpt-4.1-mini', got '{called_model}'. "
        "generate_plain() must use evaluation_model, not active_model_key."
    )


def test_generate_plain_falls_back_to_active_model_when_evaluation_model_is_none():
    """
    When evaluation_model is explicitly set to None, generate_plain() must fall
    back to active_model_key.  This is the opt-out path for operators who
    intentionally want the same model for both benchmarking and evaluation.
    """
    config = _make_config(benchmarked_model="gpt-4-1-mini", evaluation_model=None)

    with (
        patch("memorch.llm_orchestrator.MemoryProcessor"),
        patch("memorch.llm_orchestrator.litellm.completion") as mock_completion,
    ):
        mock_completion.return_value = MagicMock()
        from memorch.llm_orchestrator import LLMOrchestrator

        orch = LLMOrchestrator(config=config)
        orch.generate_plain(input_messages=[{"role": "user", "content": "test"}])

    called_model = mock_completion.call_args.kwargs.get("model")
    assert called_model == "sap/gpt-4.1-mini"


def test_generate_plain_raises_when_evaluation_model_not_in_registry():
    """
    generate_plain() must raise a ValueError with a clear message when
    evaluation_model is set to a key that does not exist in the model registry.
    A silent fallback would route evaluation calls to the wrong model without
    any indication that the config is misconfigured.
    """
    config = _make_config(evaluation_model="nonexistent-model")

    with (
        patch("memorch.llm_orchestrator.MemoryProcessor"),
        patch("memorch.llm_orchestrator.litellm.completion"),
    ):
        from memorch.llm_orchestrator import LLMOrchestrator

        orch = LLMOrchestrator(config=config)

        with pytest.raises(ValueError, match="nonexistent-model"):
            orch.generate_plain(input_messages=[{"role": "user", "content": "test"}])


def test_generate_plain_drops_caller_supplied_model_kwarg():
    """
    Any 'model' kwarg passed by the caller must be silently dropped.
    The caller (SAPGPTModel._predict) passes model=self.model_name as a kwarg,
    but generate_plain() must ignore it and always derive the model from config.
    This prevents callers from accidentally overriding the evaluation model.
    """
    config = _make_config(benchmarked_model="qwen35", evaluation_model="gpt-4-1-mini")

    with (
        patch("memorch.llm_orchestrator.MemoryProcessor"),
        patch("memorch.llm_orchestrator.litellm.completion") as mock_completion,
    ):
        mock_completion.return_value = MagicMock()
        from memorch.llm_orchestrator import LLMOrchestrator

        orch = LLMOrchestrator(config=config)
        # Caller passes model="qwen35" as if it were routing to the benchmarked model
        orch.generate_plain(
            input_messages=[{"role": "user", "content": "test"}],
            model="qwen35",
        )

    called_model = mock_completion.call_args.kwargs.get("model")
    assert called_model == "sap/gpt-4.1-mini", (
        "Caller-supplied model kwarg must be dropped; evaluation_model from config must win."
    )
