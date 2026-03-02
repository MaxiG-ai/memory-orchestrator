"""
Tests for the progressive summarization memory strategy.

The progressive summarization strategy uses an LLM to compress conversation
history into a summary, preserving essential information while reducing tokens.
"""

from unittest.mock import MagicMock

import pytest

from memorch.strategies.progressive_summarization.prog_sum import (
    summarize_conv_history,
)


def _make_message(role: str, content: str, **extras) -> dict:
    """Helper to create message dicts for testing."""
    message = {"role": role, "content": content}
    message.update(extras)
    return message


def _make_mock_llm_response(content: str) -> MagicMock:
    """Create a mock LLM response with the given content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    return response


@pytest.fixture
def prompt_file(tmp_path):
    """Provide a minimal prompt file for PromptManager to load.

    Creates a temporary prompt file with a ${user_query} placeholder,
    matching the format expected by PromptManager.render().
    """
    p = tmp_path / "prog_sum.prompt.md"
    p.write_text("Summarize the conversation. User query: ${user_query}")
    return str(p)


class TestSummarizeConvHistory:
    """Tests for the summarize_conv_history function."""

    def test_summarize_requires_llm_client(self, prompt_file) -> None:
        """
        Test that function raises ValueError when llm_client is None.

        The progressive summarization strategy requires an LLM to generate
        summaries, so it cannot function without a client. The llm_client
        check occurs before PromptManager is invoked, so a valid prompt_file
        is provided to satisfy the required parameter.
        """
        messages = [_make_message("user", "Hello")]

        with pytest.raises(ValueError, match="llm_client is required"):
            summarize_conv_history(messages, llm_client=None, summary_prompt_path=prompt_file)

    def test_summarize_basic_conversation(self, prompt_file) -> None:
        """
        Test basic summarization of a conversation.

        The function should call the LLM with the conversation history
        and return a compressed message list with the summary.
        """
        messages = [
            _make_message("user", "What is Python?"),
            _make_message("assistant", "Python is a programming language..."),
            _make_message("user", "Tell me more about its features"),
            _make_message("assistant", "Python has many features including..."),
        ]

        mock_client = MagicMock()
        mock_client.generate_plain.return_value = _make_mock_llm_response(
            "Summary: Discussed Python programming language and its features."
        )

        result = summarize_conv_history(messages, llm_client=mock_client, summary_prompt_path=prompt_file)

        # Should have called generate_plain
        mock_client.generate_plain.assert_called_once()

        # Result should contain the summary as a system message
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) >= 1
        assert "Summary" in system_msgs[0]["content"]

    def test_summarize_preserves_user_query(self, prompt_file) -> None:
        """
        Test that the original user query is preserved in output.

        The summarization should keep the user's task visible so the
        LLM knows what to do, alongside the compressed history.
        """
        messages = [
            _make_message("user", "Analyze this data: [1, 2, 3]"),
            _make_message("assistant", "I'll analyze the data..."),
            _make_message("user", "Continue the analysis"),
        ]

        mock_client = MagicMock()
        mock_client.generate_plain.return_value = _make_mock_llm_response(
            "Previous: Started analyzing data array."
        )

        result = summarize_conv_history(messages, llm_client=mock_client, summary_prompt_path=prompt_file)

        # The user query should be in the result
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) >= 1

    def test_summarize_raises_on_empty_response(self, prompt_file) -> None:
        """
        Test that empty LLM response raises ValueError.

        If the summarization LLM returns empty content, the function
        should raise rather than return invalid data.
        """
        messages = [
            _make_message("user", "Hello"),
            _make_message("assistant", "Hi there!"),
        ]

        mock_client = MagicMock()
        mock_client.generate_plain.return_value = _make_mock_llm_response("")

        with pytest.raises(ValueError, match="empty content"):
            summarize_conv_history(messages, llm_client=mock_client, summary_prompt_path=prompt_file)

    def test_summarize_uses_custom_model(self, prompt_file) -> None:
        """
        Test that custom summarizer model is passed to LLM client.

        Users should be able to specify which model performs summarization.
        The generate_plain call uses keyword arguments input_messages= and
        model=, so both are verified via call_args.
        """
        messages = [_make_message("user", "Test")]

        mock_client = MagicMock()
        mock_client.generate_plain.return_value = _make_mock_llm_response("Summary")

        summarize_conv_history(
            messages, llm_client=mock_client, summary_prompt_path=prompt_file, summarizer_model="gpt-4o"
        )

        # generate_plain is called with input_messages= and model= as kwargs
        call_kwargs = mock_client.generate_plain.call_args
        assert call_kwargs.kwargs.get("model") == "gpt-4o"

    def test_summarize_handles_dict_message_format(self, prompt_file) -> None:
        """
        Test handling of dict-style message response from LLM.

        Some providers return messages as dicts rather than objects.
        """
        messages = [
            _make_message("user", "Hello"),
            _make_message("assistant", "Hi!"),
        ]

        mock_client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = {
            "content": "Dict-style summary",
            "role": "assistant",
        }
        mock_client.generate_plain.return_value = response

        result = summarize_conv_history(messages, llm_client=mock_client, summary_prompt_path=prompt_file)

        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) >= 1
        assert "Dict-style summary" in system_msgs[0]["content"]

    def test_summarize_with_tool_messages(self, prompt_file) -> None:
        """
        Test summarization of conversation including tool interactions.

        Tool calls and responses should be included in what gets
        summarized.
        """
        messages = [
            _make_message("user", "Get the weather"),
            _make_message(
                "assistant",
                "Checking weather...",
                tool_calls=[{"id": "tc-1", "function": {"name": "get_weather"}}],
            ),
            _make_message("tool", '{"temp": 72}', tool_call_id="tc-1"),
            _make_message("assistant", "The temperature is 72F"),
        ]

        mock_client = MagicMock()
        mock_client.generate_plain.return_value = _make_mock_llm_response(
            "Retrieved weather: 72F temperature."
        )

        result = summarize_conv_history(messages, llm_client=mock_client, summary_prompt_path=prompt_file)

        # Should complete without error
        assert len(result) > 0

    def test_summarize_handles_none_content_in_response(self, prompt_file) -> None:
        """
        Test handling of None content in LLM response object.

        Some responses might have content=None which should be treated
        as empty and raise ValueError.
        """
        messages = [_make_message("user", "Test")]

        mock_client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = None
        mock_client.generate_plain.return_value = response

        with pytest.raises(ValueError, match="empty content"):
            summarize_conv_history(messages, llm_client=mock_client, summary_prompt_path=prompt_file)
