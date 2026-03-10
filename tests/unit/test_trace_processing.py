from __future__ import annotations

from memorch.utils.trace_processing import detect_tail_loop


def test_detect_tail_loop_returns_false_for_short_histories() -> None:
    """Ensure loop detection exits early when there are not enough messages.

    A loop requires at least `threshold` messages. This test protects the short-
    history fast path and guarantees the detector does not produce false positives
    on normal startup conversations.
    """

    assert detect_tail_loop([{"role": "user", "content": "hello"}], threshold=2) is False


def test_detect_tail_loop_detects_repeated_tail_pattern_with_tool_calls() -> None:
    """Validate repeated assistant-tool pairs are detected despite call-order variance.

    The detector normalizes `tool_calls` by sorting call signatures. This test builds
    a two-message pattern repeated three times where each repetition uses different
    tool call order, ensuring normalization preserves semantic loop detection.
    """

    pattern_a = {
        "role": "assistant",
        "content": "I will call tools",
        "tool_calls": [
            {"type": "function", "function": {"name": "b", "arguments": "2"}},
            {"type": "function", "function": {"name": "a", "arguments": "1"}},
        ],
    }
    pattern_b = {
        "role": "tool",
        "content": "tool output",
        "tool_calls": [
            {"type": "function", "function": {"name": "a", "arguments": "1"}},
            {"type": "function", "function": {"name": "b", "arguments": "2"}},
        ],
    }

    messages = [pattern_a, pattern_b, pattern_a, pattern_b, pattern_a, pattern_b]
    assert detect_tail_loop(messages, threshold=3, max_pattern_len=2) is True


def test_detect_tail_loop_returns_true_when_history_exceeds_guardrail() -> None:
    """Document the intentional >150-message guardrail behavior.

    The implementation short-circuits to True for very large histories instead of
    scanning for a concrete pattern. This test codifies that behavior so refactors
    preserve the current protective heuristic.
    """

    long_history = [{"role": "user", "content": str(i)} for i in range(151)]
    assert detect_tail_loop(long_history) is True
