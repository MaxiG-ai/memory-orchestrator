"""
Tests for the Open Responses / Chat-Completions format translator.

This module tests the bidirectional translation between OpenAI's chat-completions
format (used internally by LLMOrchestrator) and the Open Responses format
(used by the external API).

The translator handles:
- User/system messages -> MessageItem with InputTextContent
- Assistant messages -> MessageItem with OutputTextContent
- Tool calls (in assistant messages) -> FunctionCallItem
- Tool responses -> FunctionCallOutputItem

Key edge cases tested:
- Empty/None content handling
- String vs array content formats
- Multiple consecutive function calls
- Round-trip conversion consistency
"""


from src.memorch.api.translator import (
    chat_completions_to_items,
    items_to_chat_completions,
)
from src.memorch.api.models import (
    MessageItem,
    FunctionCallItem,
    FunctionCallOutputItem,
    InputTextContent,
    OutputTextContent,
    MessageRole,
    ItemStatus,
)


class TestChatCompletionsToItems:
    """Tests for converting chat-completions format to Open Responses items."""

    def test_user_message_converts_to_message_item(self) -> None:
        """
        Test that a simple user message converts to a MessageItem.

        A user message in chat-completions format:
            {"role": "user", "content": "Hello"}
        Should become a MessageItem with InputTextContent.
        """
        messages = [{"role": "user", "content": "Hello"}]

        items = chat_completions_to_items(messages)

        assert len(items) == 1
        assert isinstance(items[0], MessageItem)
        assert items[0].role == MessageRole.user
        assert len(items[0].content) == 1
        assert isinstance(items[0].content[0], InputTextContent)
        assert items[0].content[0].text == "Hello"

    def test_system_message_converts_to_message_item(self) -> None:
        """
        Test that a system message converts to a MessageItem with InputTextContent.

        System messages use the same InputTextContent format as user messages
        since they are both inputs to the model.
        """
        messages = [{"role": "system", "content": "You are helpful."}]

        items = chat_completions_to_items(messages)

        assert len(items) == 1
        assert isinstance(items[0], MessageItem)
        assert items[0].role == MessageRole.system
        assert isinstance(items[0].content[0], InputTextContent)
        assert items[0].content[0].text == "You are helpful."

    def test_assistant_message_converts_to_message_item(self) -> None:
        """
        Test that an assistant message converts to MessageItem with OutputTextContent.

        Assistant messages are outputs from the model, so they use OutputTextContent
        to distinguish them from input content.
        """
        messages = [{"role": "assistant", "content": "Hi there!"}]

        items = chat_completions_to_items(messages)

        assert len(items) == 1
        assert isinstance(items[0], MessageItem)
        assert items[0].role == MessageRole.assistant
        assert len(items[0].content) == 1
        assert isinstance(items[0].content[0], OutputTextContent)
        assert items[0].content[0].text == "Hi there!"

    def test_assistant_with_tool_calls_creates_function_call_items(self) -> None:
        """
        Test that tool_calls in an assistant message become separate FunctionCallItems.

        In chat-completions format, tool calls are embedded in the assistant message.
        In Open Responses format, they are separate items following the message.
        """
        messages = [
            {
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            }
        ]

        items = chat_completions_to_items(messages)

        # Should have 2 items: MessageItem + FunctionCallItem
        assert len(items) == 2
        assert isinstance(items[0], MessageItem)
        assert items[0].content[0].text == "Let me check that."
        assert isinstance(items[1], FunctionCallItem)
        assert items[1].call_id == "call_abc123"
        assert items[1].name == "get_weather"
        assert items[1].arguments == '{"city": "NYC"}'

    def test_tool_message_converts_to_function_call_output_item(self) -> None:
        """
        Test that tool role messages convert to FunctionCallOutputItem.

        Tool messages in chat-completions are responses from function execution.
        They link back to the original call via tool_call_id.
        """
        messages = [
            {"role": "tool", "tool_call_id": "call_abc123", "content": "Sunny, 72F"}
        ]

        items = chat_completions_to_items(messages)

        assert len(items) == 1
        assert isinstance(items[0], FunctionCallOutputItem)
        assert items[0].call_id == "call_abc123"
        assert items[0].output == "Sunny, 72F"

    def test_multiple_tool_calls_in_single_message(self) -> None:
        """
        Test that multiple tool_calls create multiple FunctionCallItems.

        An assistant message can request multiple tool calls simultaneously.
        Each should become a separate FunctionCallItem.
        """
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "fn1", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "fn2", "arguments": "{}"},
                    },
                ],
            }
        ]

        items = chat_completions_to_items(messages)

        # MessageItem (with empty content) + 2 FunctionCallItems
        assert len(items) == 3
        assert isinstance(items[0], MessageItem)
        assert isinstance(items[1], FunctionCallItem)
        assert isinstance(items[2], FunctionCallItem)
        assert items[1].call_id == "call_1"
        assert items[2].call_id == "call_2"

    def test_empty_content_handled_gracefully(self) -> None:
        """
        Test that empty string content creates an empty content list.

        Some messages may have content="" which should result in no
        content items rather than a content item with empty text.
        """
        messages = [{"role": "user", "content": ""}]

        items = chat_completions_to_items(messages)

        assert len(items) == 1
        assert isinstance(items[0], MessageItem)
        assert len(items[0].content) == 0

    def test_none_content_handled_gracefully(self) -> None:
        """
        Test that None content creates an empty content list.

        Assistant messages with tool calls often have content=None.
        """
        messages = [{"role": "assistant", "content": None}]

        items = chat_completions_to_items(messages)

        assert len(items) == 1
        assert isinstance(items[0], MessageItem)
        assert len(items[0].content) == 0

    def test_full_conversation_converts_correctly(self) -> None:
        """
        Test conversion of a complete conversation with multiple message types.

        Verifies the translator handles a realistic conversation flow:
        system -> user -> assistant -> tool_calls -> tool_response -> assistant
        """
        messages = [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "What's the weather in NYC?"},
            {
                "role": "assistant",
                "content": "I'll check that for you.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 72F"},
            {"role": "assistant", "content": "It's sunny and 72F in NYC!"},
        ]

        items = chat_completions_to_items(messages)

        # system + user + assistant + function_call + function_output + assistant
        assert len(items) == 6
        assert items[0].role == MessageRole.system
        assert items[1].role == MessageRole.user
        assert items[2].role == MessageRole.assistant
        assert isinstance(items[3], FunctionCallItem)
        assert isinstance(items[4], FunctionCallOutputItem)
        assert items[5].role == MessageRole.assistant


class TestItemsToChatCompletions:
    """Tests for converting Open Responses items to chat-completions format."""

    def test_user_message_item_converts_to_message(self) -> None:
        """
        Test that a user MessageItem converts to a chat-completions message.

        The InputTextContent text should become the content string.
        """
        items = [
            MessageItem(
                role=MessageRole.user,
                content=[InputTextContent(text="Hello")],
            )
        ]

        messages = items_to_chat_completions(items)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_assistant_message_item_converts_to_message(self) -> None:
        """
        Test that an assistant MessageItem converts to a chat-completions message.

        OutputTextContent should become the content string in the message.
        """
        items = [
            MessageItem(
                role=MessageRole.assistant,
                content=[OutputTextContent(text="Hi there!")],
            )
        ]

        messages = items_to_chat_completions(items)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hi there!"

    def test_function_call_item_attaches_to_preceding_assistant(self) -> None:
        """
        Test that FunctionCallItem attaches to the preceding assistant message.

        In chat-completions format, tool_calls are part of the assistant message.
        The translator should group consecutive function calls into the assistant
        message's tool_calls array.
        """
        items = [
            MessageItem(
                role=MessageRole.assistant,
                content=[OutputTextContent(text="Let me check.")],
            ),
            FunctionCallItem(
                call_id="call_1",
                name="get_weather",
                arguments='{"city": "NYC"}',
                status=ItemStatus.completed,
            ),
        ]

        messages = items_to_chat_completions(items)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Let me check."
        assert "tool_calls" in messages[0]
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[0]["tool_calls"][0]["id"] == "call_1"
        assert messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_function_call_creates_assistant_message_if_none_exists(self) -> None:
        """
        Test that FunctionCallItem creates an assistant message if there's no preceding one.

        If a FunctionCallItem appears without a preceding assistant message,
        the translator should create an assistant message with content=None.
        """
        items = [
            FunctionCallItem(
                call_id="call_1",
                name="fn",
                arguments="{}",
                status=ItemStatus.completed,
            ),
        ]

        messages = items_to_chat_completions(items)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] is None
        assert len(messages[0]["tool_calls"]) == 1

    def test_function_call_output_converts_to_tool_message(self) -> None:
        """
        Test that FunctionCallOutputItem converts to a tool role message.

        The output becomes the content, and call_id becomes tool_call_id.
        """
        items = [
            FunctionCallOutputItem(
                call_id="call_1",
                output="Result data",
                status=ItemStatus.completed,
            ),
        ]

        messages = items_to_chat_completions(items)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_1"
        assert messages[0]["content"] == "Result data"

    def test_multiple_consecutive_function_calls(self) -> None:
        """
        Test that multiple consecutive FunctionCallItems attach to the same assistant message.

        When multiple function calls follow an assistant message (or each other),
        they should all be grouped into one assistant message's tool_calls array.
        """
        items = [
            MessageItem(
                role=MessageRole.assistant,
                content=[],  # Empty content (tool-call-only response)
            ),
            FunctionCallItem(
                call_id="call_1",
                name="fn1",
                arguments="{}",
                status=ItemStatus.completed,
            ),
            FunctionCallItem(
                call_id="call_2",
                name="fn2",
                arguments="{}",
                status=ItemStatus.completed,
            ),
        ]

        messages = items_to_chat_completions(items)

        assert len(messages) == 1
        assert len(messages[0]["tool_calls"]) == 2
        assert messages[0]["tool_calls"][0]["id"] == "call_1"
        assert messages[0]["tool_calls"][1]["id"] == "call_2"

    def test_empty_content_list_results_in_none_content(self) -> None:
        """
        Test that MessageItem with empty content list becomes content=None.

        This handles the case of assistant messages that only have tool calls.
        """
        items = [
            MessageItem(role=MessageRole.assistant, content=[]),
        ]

        messages = items_to_chat_completions(items)

        assert messages[0]["content"] is None

    def test_multiple_content_items_concatenated(self) -> None:
        """
        Test that multiple content items in a MessageItem are concatenated.

        Open Responses allows multiple content objects. When converting to
        chat-completions, these should be joined with newlines.
        """
        items = [
            MessageItem(
                role=MessageRole.user,
                content=[
                    InputTextContent(text="First part."),
                    InputTextContent(text="Second part."),
                ],
            ),
        ]

        messages = items_to_chat_completions(items)

        assert messages[0]["content"] == "First part.\nSecond part."

    def test_full_conversation_round_trip(self) -> None:
        """
        Test that a complete conversation converts back correctly.

        This integration test verifies the full flow:
        items -> chat_completions -> items (approximately equal)
        """
        items = [
            MessageItem(
                role=MessageRole.system,
                content=[InputTextContent(text="System prompt")],
            ),
            MessageItem(
                role=MessageRole.user, content=[InputTextContent(text="User question")]
            ),
            MessageItem(
                role=MessageRole.assistant,
                content=[OutputTextContent(text="I'll help")],
            ),
            FunctionCallItem(
                call_id="call_1",
                name="tool",
                arguments='{"a": 1}',
                status=ItemStatus.completed,
            ),
            FunctionCallOutputItem(
                call_id="call_1", output="tool result", status=ItemStatus.completed
            ),
            MessageItem(
                role=MessageRole.assistant,
                content=[OutputTextContent(text="Here's the answer")],
            ),
        ]

        messages = items_to_chat_completions(items)

        # system + user + assistant_with_tool_calls + tool + assistant
        assert len(messages) == 5
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert "tool_calls" in messages[2]
        assert messages[3]["role"] == "tool"
        assert messages[4]["role"] == "assistant"


class TestDictItemHandling:
    """Tests for handling both dict and Pydantic model items."""

    def test_items_to_chat_handles_dict_items(self) -> None:
        """
        Test that items_to_chat_completions handles dict-format items.

        Items may come as raw dicts from JSON deserialization.
        The translator should handle both dicts and Pydantic models.
        """
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello from dict"}],
            },
        ]

        messages = items_to_chat_completions(items)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello from dict"

    def test_items_to_chat_handles_dict_function_call(self) -> None:
        """
        Test that dict-format FunctionCallItems are handled correctly.

        Verifies function call dicts attach properly to assistant messages.
        """
        items = [
            {"type": "message", "role": "assistant", "content": []},
            {
                "type": "function_call",
                "call_id": "call_dict",
                "name": "dict_fn",
                "arguments": "{}",
            },
        ]

        messages = items_to_chat_completions(items)

        assert len(messages) == 1
        assert messages[0]["tool_calls"][0]["id"] == "call_dict"

    def test_items_to_chat_handles_dict_function_output(self) -> None:
        """
        Test that dict-format FunctionCallOutputItems are handled correctly.
        """
        items = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "dict output",
            },
        ]

        messages = items_to_chat_completions(items)

        assert messages[0]["role"] == "tool"
        assert messages[0]["content"] == "dict output"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_messages_list(self) -> None:
        """
        Test that empty input returns empty output.
        """
        assert chat_completions_to_items([]) == []
        assert items_to_chat_completions([]) == []

    def test_function_call_after_non_assistant_message(self) -> None:
        """
        Test FunctionCallItem after a non-assistant message creates new assistant message.

        If a user message precedes a function call, a new assistant message
        should be created to hold the tool_calls.
        """
        items = [
            MessageItem(
                role=MessageRole.user, content=[InputTextContent(text="Question")]
            ),
            FunctionCallItem(
                call_id="call_1", name="fn", arguments="{}", status=ItemStatus.completed
            ),
        ]

        messages = items_to_chat_completions(items)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] is None
        assert len(messages[1]["tool_calls"]) == 1

    def test_string_content_in_open_responses_format(self) -> None:
        """
        Test handling of string content (shorthand) in item dicts.

        Some implementations may use string content directly instead of
        an array of content objects. The translator should handle this.
        """
        items = [
            {
                "type": "message",
                "role": "user",
                "content": "Direct string content",  # String instead of array
            },
        ]

        messages = items_to_chat_completions(items)

        assert messages[0]["content"] == "Direct string content"
