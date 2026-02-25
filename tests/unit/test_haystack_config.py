"""Tests for haystack experiment configuration: configurable message limits and thresholds."""

from unittest.mock import patch, MagicMock
import pytest

from memorch.utils.config import ExperimentConfig


# --- Minimal valid config dict reused across tests ---


def _base_config(**overrides) -> dict:
    """Return a minimal valid ExperimentConfig dict with optional overrides."""
    base = {
        "experiment_name": "test",
        "results_dir": "results",
        "log_dir": "logs",
        "logging_level": "WARNING",
        "weave_logging": False,
        "input_file": "data.jsonl",
        "enabled_models": ["gpt-test"],
        "enabled_memory_methods": ["no_strategy"],
        "compact_thresholds": [5000],
        "memory_strategies": {"no_strategy": {"type": "no_strategy"}},
        "model_registry": {
            "gpt-test": {
                "litellm_name": "openai/gpt-test",
                "context_window": 128000,
                "provider": "test",
            }
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ExperimentConfig field defaults and validation
# ---------------------------------------------------------------------------


def test_max_messages_default_is_40():
    """Verify that ExperimentConfig defaults max_messages_after_compression to 40.

    Backward compatibility: existing configs that don't set this field must
    continue to enforce the original hard limit of 40 messages.
    """
    cfg = ExperimentConfig.model_validate(_base_config())
    assert cfg.max_messages_after_compression == 40


def test_max_messages_configurable():
    """Verify that max_messages_after_compression can be set to a custom value.

    For haystack experiments with large injected contexts, conversations may
    legitimately contain 60-100+ messages.  The limit must be raisable.
    """
    cfg = ExperimentConfig.model_validate(
        _base_config(max_messages_after_compression=200)
    )
    assert cfg.max_messages_after_compression == 200


def test_max_messages_none_accepted():
    """Verify that None disables the post-compression message limit entirely.

    Setting max_messages_after_compression=None means 'no upper bound',
    useful when the experiment itself controls message count.
    """
    cfg = ExperimentConfig.model_validate(
        _base_config(max_messages_after_compression=None)
    )
    assert cfg.max_messages_after_compression is None


def test_haystack_thresholds_default_none():
    """Verify haystack_thresholds defaults to None when not specified.

    None indicates that no haystack experiment is configured; the field
    should be entirely optional.
    """
    cfg = ExperimentConfig.model_validate(_base_config())
    assert cfg.haystack_thresholds is None


def test_haystack_thresholds_accepts_list():
    """Verify haystack_thresholds accepts a list of integer token targets.

    Typical haystack experiments sweep across several context sizes,
    e.g. [20_000, 40_000, 60_000, 80_000, 100_000].
    """
    targets = [20_000, 40_000, 60_000, 80_000, 100_000]
    cfg = ExperimentConfig.model_validate(_base_config(haystack_thresholds=targets))
    assert cfg.haystack_thresholds == targets


# ---------------------------------------------------------------------------
# Integration: generate_with_memory_applied respects the configured limit
# ---------------------------------------------------------------------------


def test_generate_respects_configured_limit():
    """Verify generate_with_memory_applied asserts against the configured limit.

    When max_messages_after_compression is set to 5, a compressed view of 10
    messages must trigger an AssertionError.  We mock litellm.completion and
    the memory processor so no real API calls occur.
    """
    from memorch.llm_orchestrator import LLMOrchestrator

    cfg = ExperimentConfig.model_validate(
        _base_config(max_messages_after_compression=5)
    )

    # Patch __init__ to inject our custom config without loading config files
    with patch.object(LLMOrchestrator, "__init__", lambda self, *a, **kw: None):
        orch = LLMOrchestrator.__new__(LLMOrchestrator)
        orch.cfg = cfg
        orch.active_model_key = "gpt-test"
        orch.active_memory_key = "no_strategy"
        orch.active_compact_threshold = None
        orch._compressed_trace_buffer = []
        orch._trace_step_counter = 0

        # Build a mock memory processor that returns too many messages
        mock_processor = MagicMock()
        # 10 messages > configured limit of 5
        too_many = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        mock_processor.apply_strategy.return_value = (too_many, 500)
        orch.memory_processor = mock_processor

        with patch("memorch.llm_orchestrator.get_token_count", return_value=100):
            with pytest.raises(AssertionError, match="Too many messages"):
                orch.generate_with_memory_applied(
                    [{"role": "user", "content": "hello"}]
                )


def test_generate_skips_check_when_none():
    """Verify generate_with_memory_applied skips the assertion when limit is None.

    When max_messages_after_compression is None, even a large number of
    compressed messages should pass through without triggering the guard.
    We mock litellm.completion to return a valid-looking response.
    """
    from memorch.llm_orchestrator import LLMOrchestrator

    cfg = ExperimentConfig.model_validate(
        _base_config(max_messages_after_compression=None)
    )

    with patch.object(LLMOrchestrator, "__init__", lambda self, *a, **kw: None):
        orch = LLMOrchestrator.__new__(LLMOrchestrator)
        orch.cfg = cfg
        orch.active_model_key = "gpt-test"
        orch.active_memory_key = "no_strategy"
        orch.active_compact_threshold = None
        orch._compressed_trace_buffer = []
        orch._trace_step_counter = 0

        # 100 messages — well above the old hard limit of 40
        many_msgs = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
        mock_processor = MagicMock()
        mock_processor.apply_strategy.return_value = (many_msgs, 5000)
        orch.memory_processor = mock_processor

        # Mock litellm.completion to return a fake response
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "ok"

        with patch("memorch.llm_orchestrator.get_token_count", return_value=100):
            with patch("memorch.llm_orchestrator.litellm") as mock_litellm:
                mock_litellm.completion.return_value = fake_response
                # Should NOT raise — the limit check is disabled
                result = orch.generate_with_memory_applied(many_msgs)
                assert result is fake_response
