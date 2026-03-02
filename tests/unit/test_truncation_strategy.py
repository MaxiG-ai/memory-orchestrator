"""
Tests for the truncation memory strategy.

The truncation strategy removes conversation history between the user query
and the last n tool interactions, keeping only the essential context for the LLM.
The parameter keep_last_n_tool_interactions (default=1) controls how many
tool interactions are retained.
"""

import pytest

from memorch.strategies.truncation.truncation import truncate_messages


def _make_message(role: str, content: str, **extras) -> dict:
    """Helper to create message dicts for testing."""
    message = {"role": role, "content": content}
    message.update(extras)
    return message


def _make_tool_turn(assistant_content: str, tc_id: str, tool_name: str, tool_result: str) -> list:
    """Return [assistant-with-tool_calls, tool-response] for a single tool turn."""
    return [
        _make_message(
            "assistant",
            assistant_content,
            tool_calls=[{"id": tc_id, "type": "function", "function": {"name": tool_name}}],
        ),
        _make_message("tool", tool_result, tool_call_id=tc_id),
    ]


class TestTruncateMessages:
    """Tests for the truncate_messages function."""

    # ------------------------------------------------------------------ #
    # Default behaviour: keep_last_n_tool_interactions=1                  #
    # ------------------------------------------------------------------ #

    def test_truncate_removes_intermediate_history(self) -> None:
        """
        Test that intermediate conversation history is removed with default n=1.

        The truncation strategy should keep the first user message and the
        last tool interaction, discarding everything in between to reduce
        context size while preserving essential information.
        """
        messages = [
            _make_message("system", "System prompt"),
            _make_message("user", "First question - this is the task"),
            _make_message("assistant", "First answer - should be removed"),
            _make_message("user", "Follow-up question - should be removed"),
            _make_message("assistant", "Second answer - should be removed"),
            *_make_tool_turn("Calling tool", "tc-1", "get_data", '{"result": "data"}'),
        ]

        result = truncate_messages(messages)

        # Should have: first user message + tool interaction (assistant + tool)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "First question - this is the task"
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert result[2]["role"] == "tool"

    def test_truncate_no_tool_interaction(self) -> None:
        """
        Test truncation when there's no tool interaction at the end.

        When there's no tool interaction, the strategy keeps only the user
        query since there's no tool context to preserve. The assistant
        response is considered intermediate history and is discarded.
        """
        messages = [
            _make_message("user", "Simple question"),
            _make_message("assistant", "Simple answer"),
        ]

        result = truncate_messages(messages)

        # Only user query is kept (no tool interaction to preserve)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Simple question"

    def test_truncate_empty_messages(self) -> None:
        """
        Test truncation with empty message list.

        Should handle edge case gracefully without raising exceptions.
        """
        result = truncate_messages([])

        assert result == []

    def test_truncate_only_user_message(self) -> None:
        """
        Test truncation with only a user message.

        The simplest valid conversation should be preserved intact.
        """
        messages = [_make_message("user", "Hello")]

        result = truncate_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_truncate_preserves_tool_call_details(self) -> None:
        """
        Test that tool call details are preserved in truncated output.

        The tool_calls structure and tool response content should be
        kept intact for the LLM to process correctly.
        """
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "test"}'},
            }
        ]
        messages = [
            _make_message("user", "Search for test"),
            _make_message("assistant", "Searching...", tool_calls=tool_calls),
            _make_message(
                "tool", '{"results": ["item1", "item2"]}', tool_call_id="call_abc123"
            ),
        ]

        result = truncate_messages(messages)

        # Find the assistant message with tool_calls
        assistant_msg = next((m for m in result if m.get("tool_calls")), None)
        assert assistant_msg is not None
        assert assistant_msg["tool_calls"] == tool_calls

        # Find the tool response
        tool_msg = next((m for m in result if m["role"] == "tool"), None)
        assert tool_msg is not None
        assert tool_msg["tool_call_id"] == "call_abc123"

    def test_truncate_multiple_parallel_tool_calls(self) -> None:
        """
        Test truncation preserves multiple parallel tool calls within a single turn.

        When an assistant makes multiple tool calls in one turn, all
        should be preserved in the truncated output.
        """
        tool_calls = [
            {"id": "tc-1", "type": "function", "function": {"name": "get_weather"}},
            {"id": "tc-2", "type": "function", "function": {"name": "get_time"}},
        ]
        messages = [
            _make_message("user", "What's the weather and time?"),
            _make_message("assistant", "Getting both...", tool_calls=tool_calls),
            _make_message("tool", '{"temp": 72}', tool_call_id="tc-1"),
            _make_message("tool", '{"time": "3pm"}', tool_call_id="tc-2"),
        ]

        result = truncate_messages(messages)

        # Should have user + assistant + 2 tool responses
        assert len(result) == 4
        tool_responses = [m for m in result if m["role"] == "tool"]
        assert len(tool_responses) == 2

    def test_truncate_with_system_message(self) -> None:
        """
        Test that system messages are handled correctly.

        System messages are not user messages, so they are not preserved by
        the strategy. Only the first user message is kept as the query.
        """
        messages = [
            _make_message("system", "You are a helpful assistant"),
            _make_message("user", "Hello"),
            _make_message("assistant", "Hi there!"),
        ]

        result = truncate_messages(messages)

        # The first user message should be preserved
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) >= 1

    def test_truncate_long_conversation(self) -> None:
        """
        Test truncation on a longer conversation with multiple turns.

        This simulates a real-world scenario where context has grown
        large and needs to be compressed.
        """
        messages = [
            _make_message("system", "System prompt"),
            _make_message("user", "Initial task: analyze this data"),
            # Many intermediate turns
            _make_message("assistant", "I'll start by..."),
            _make_message("user", "Good, continue"),
            _make_message("assistant", "Next step..."),
            _make_message("user", "What about X?"),
            _make_message("assistant", "For X, I'll..."),
            _make_message("user", "Ok proceed"),
            _make_message("assistant", "Analyzing..."),
            # Final tool interaction
            *_make_tool_turn("Running analysis", "tc-final", "analyze", '{"analysis": "complete"}'),
        ]

        result = truncate_messages(messages)

        # Should be much shorter than original
        assert len(result) < len(messages)
        # Should preserve the first user message (task)
        assert result[0]["role"] == "user"
        assert "Initial task" in result[0]["content"]
        # Should preserve the final tool interaction
        assert result[-1]["role"] == "tool"

    # ------------------------------------------------------------------ #
    # keep_last_n_tool_interactions parameter                             #
    # ------------------------------------------------------------------ #

    def test_keep_zero_tool_interactions_returns_only_user_query(self) -> None:
        """
        Test that keep_last_n_tool_interactions=0 returns only the user query.

        When n=0 the while-loop body never executes, so no tool interactions
        are appended and the result is exclusively the first user message.
        """
        messages = [
            _make_message("user", "Do something"),
            *_make_tool_turn("Calling tool", "tc-1", "do_it", '{"ok": true}'),
        ]

        result = truncate_messages(messages, keep_last_n_tool_interactions=0)

        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_keep_two_tool_interactions(self) -> None:
        """
        Test that keep_last_n_tool_interactions=2 retains the last two tool turns.

        With two distinct tool interactions the loop runs twice, peeling off the
        most-recent turn first, then the one before it. Both should appear in the
        result alongside the original user query.
        """
        turn1 = _make_tool_turn("First tool call", "tc-1", "step_one", '{"step": 1}')
        turn2 = _make_tool_turn("Second tool call", "tc-2", "step_two", '{"step": 2}')
        messages = [
            _make_message("user", "Run two steps"),
            _make_message("assistant", "Intermediate text - should be dropped"),
            *turn1,
            _make_message("assistant", "Intermediate text 2 - should be dropped"),
            *turn2,
        ]

        result = truncate_messages(messages, keep_last_n_tool_interactions=2)

        # user + turn2 (most recent, appended first) + turn1 (older, appended second)
        assert result[0]["role"] == "user"
        tool_call_ids = [m.get("tool_call_id") for m in result if m["role"] == "tool"]
        assert "tc-2" in tool_call_ids
        assert "tc-1" in tool_call_ids
        # Each turn contributes assistant+tool = 2 messages, plus the user query
        assert len(result) == 5

    def test_keep_more_interactions_than_available(self) -> None:
        """
        Test graceful behaviour when n exceeds the number of tool interactions.

        If keep_last_n_tool_interactions=3 but only one tool interaction exists,
        the loop terminates early once get_last_tool_interaction returns an empty
        list (causing old_messages to become empty), without raising an exception.
        Only the available interaction is included.
        """
        messages = [
            _make_message("user", "Single step task"),
            *_make_tool_turn("One tool call", "tc-only", "only_tool", '{"done": true}'),
        ]

        result = truncate_messages(messages, keep_last_n_tool_interactions=3)

        # Should not raise; result contains user + the one available tool turn
        assert result[0]["role"] == "user"
        tool_call_ids = [m.get("tool_call_id") for m in result if m["role"] == "tool"]
        assert tool_call_ids == ["tc-only"]

    def test_keep_zero_with_no_tool_interactions(self) -> None:
        """
        Test that keep_last_n_tool_interactions=0 works when there are no tool
        interactions at all.

        Both the loop condition (n=0) and the absence of tool turns cause the
        result to be just the user query.
        """
        messages = [
            _make_message("user", "Plain question"),
            _make_message("assistant", "Plain answer"),
        ]

        result = truncate_messages(messages, keep_last_n_tool_interactions=0)

        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_multiple_user_messages_keeps_only_first(self) -> None:
        """
        Test that when the trace contains multiple user messages only the first
        is used as the query.

        The function logs a warning in this situation and slices user_msgs[:1],
        so regardless of how many user turns exist the result starts with the
        very first user message.
        """
        messages = [
            _make_message("user", "Original task"),
            _make_message("assistant", "Working on it..."),
            _make_message("user", "Follow-up instruction - should be dropped"),
            *_make_tool_turn("Tool call", "tc-1", "tool", '{"r": 1}'),
        ]

        result = truncate_messages(messages)

        user_msgs_in_result = [m for m in result if m["role"] == "user"]
        assert len(user_msgs_in_result) == 1
        assert user_msgs_in_result[0]["content"] == "Original task"
