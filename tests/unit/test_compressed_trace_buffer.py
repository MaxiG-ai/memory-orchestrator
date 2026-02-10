"""
Tests for the compressed trace buffer functionality in LLMOrchestrator.

The compressed trace buffer captures memory-processed messages for each LLM call
within a session. This enables saving the actual context sent to models (with
playbooks, reasoning traces, etc.) separately from the original conversation.
"""

import pytest
from unittest.mock import MagicMock, patch

from memorch.llm_orchestrator import (
    LLMOrchestrator,
    CompressedTraceEntry,
)


@pytest.fixture
def mock_orchestrator():
    """
    Create an orchestrator with mocked config loading.

    Patches config loading to avoid file dependencies and provides
    a minimal valid configuration for testing trace buffer behavior.
    """
    with patch("memorch.llm_orchestrator.load_configs") as mock_load:
        # Create minimal mock config
        mock_cfg = MagicMock()
        mock_cfg.enabled_models = ["test-model"]
        mock_cfg.enabled_memory_methods = ["no_strategy"]
        mock_cfg.weave_deep_logging = False
        mock_cfg.memory_strategies = {"no_strategy": MagicMock()}
        mock_cfg.model_registry = {
            "test-model": MagicMock(
                litellm_name="test-model",
                api_base=None,
                api_key=None,
            )
        }
        mock_load.return_value = mock_cfg

        orchestrator = LLMOrchestrator()
        return orchestrator


class TestCompressedTraceBuffer:
    """Tests for the session-level compressed trace buffer."""

    def test_buffer_initialized_empty(self, mock_orchestrator):
        """
        Verify that a new orchestrator starts with an empty trace buffer.

        The buffer should be empty on initialization and have a step counter
        starting at 0, ready to track calls.
        """
        assert mock_orchestrator._compressed_trace_buffer == []
        assert mock_orchestrator._trace_step_counter == 0

    def test_reset_session_clears_buffer(self, mock_orchestrator):
        """
        Verify that reset_session() clears the trace buffer.

        When starting a new benchmark case, reset_session is called to clear
        memory state. This must also clear the trace buffer so that traces
        from previous cases don't leak into subsequent cases.
        """
        # Manually add an entry to simulate a previous session
        mock_orchestrator._compressed_trace_buffer.append(
            CompressedTraceEntry(
                step=1,
                input_token_count=100,
                compressed_token_count=50,
                compression_ratio=0.5,
                memory_method="test",
                compressed_messages=[{"role": "user", "content": "test"}],
            )
        )
        mock_orchestrator._trace_step_counter = 1

        # Reset session
        mock_orchestrator.reset_session()

        # Buffer should be cleared
        assert mock_orchestrator._compressed_trace_buffer == []
        assert mock_orchestrator._trace_step_counter == 0

    def test_get_compressed_trace_returns_copy(self, mock_orchestrator):
        """
        Verify that get_compressed_trace() returns a copy of the buffer.

        Returning a copy prevents external code from accidentally modifying
        the internal buffer state.
        """
        entry = CompressedTraceEntry(
            step=1,
            input_token_count=100,
            compressed_token_count=50,
            compression_ratio=0.5,
            memory_method="test",
            compressed_messages=[{"role": "user", "content": "test"}],
        )
        mock_orchestrator._compressed_trace_buffer.append(entry)

        # Get trace and verify it's a copy
        trace = mock_orchestrator.get_compressed_trace()
        assert trace == [entry]

        # Modifying returned list shouldn't affect internal buffer
        trace.clear()
        assert len(mock_orchestrator._compressed_trace_buffer) == 1

    def test_get_compressed_trace_as_dicts(self, mock_orchestrator):
        """
        Verify that get_compressed_trace_as_dicts() returns serializable dicts.

        This method is used for JSON serialization when saving compressed
        traces to disk. The output must be JSON-compatible dictionaries.
        """
        entry = CompressedTraceEntry(
            step=1,
            input_token_count=100,
            compressed_token_count=50,
            compression_ratio=0.5,
            memory_method="ace",
            compressed_messages=[
                {"role": "system", "content": "## PLAYBOOK\n..."},
                {"role": "user", "content": "Hello"},
            ],
        )
        mock_orchestrator._compressed_trace_buffer.append(entry)

        dicts = mock_orchestrator.get_compressed_trace_as_dicts()

        assert len(dicts) == 1
        assert dicts[0]["step"] == 1
        assert dicts[0]["input_token_count"] == 100
        assert dicts[0]["compressed_token_count"] == 50
        assert dicts[0]["compression_ratio"] == 0.5
        assert dicts[0]["memory_method"] == "ace"
        assert len(dicts[0]["compressed_messages"]) == 2

    def test_multiple_entries_tracked(self, mock_orchestrator):
        """
        Verify that multiple LLM calls result in multiple trace entries.

        A typical benchmark case involves multiple LLM calls (one per turn).
        Each call should create a separate entry with incrementing step numbers.
        """
        # Simulate multiple calls by adding entries manually
        for i in range(3):
            mock_orchestrator._trace_step_counter += 1
            mock_orchestrator._compressed_trace_buffer.append(
                CompressedTraceEntry(
                    step=mock_orchestrator._trace_step_counter,
                    input_token_count=100 + i * 50,
                    compressed_token_count=50 + i * 25,
                    compression_ratio=0.5,
                    memory_method="ace",
                    compressed_messages=[{"role": "user", "content": f"Turn {i + 1}"}],
                )
            )

        trace = mock_orchestrator.get_compressed_trace()
        assert len(trace) == 3
        assert [e.step for e in trace] == [1, 2, 3]
        assert trace[0].input_token_count == 100
        assert trace[1].input_token_count == 150
        assert trace[2].input_token_count == 200


class TestCompressedTraceEntryDataclass:
    """Tests for the CompressedTraceEntry dataclass structure."""

    def test_entry_fields(self):
        """
        Verify CompressedTraceEntry has all required fields.

        The entry must capture: step number, token counts (input/compressed),
        compression ratio, memory method used, and the actual compressed messages.
        """
        entry = CompressedTraceEntry(
            step=1,
            input_token_count=1000,
            compressed_token_count=500,
            compression_ratio=0.5,
            memory_method="progressive_summarization",
            compressed_messages=[
                {"role": "system", "content": "Summary: ..."},
                {"role": "user", "content": "Question"},
            ],
        )

        assert entry.step == 1
        assert entry.input_token_count == 1000
        assert entry.compressed_token_count == 500
        assert entry.compression_ratio == 0.5
        assert entry.memory_method == "progressive_summarization"
        assert len(entry.compressed_messages) == 2
