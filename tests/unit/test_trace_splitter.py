from memorch.utils.split_trace import (
    get_user_message,
    get_first_user_text,
    get_last_tool_interaction,
    extract_tool_outputs,
)


def _make_message(role: str, content: str, **extras) -> dict:
    message = {
        "role": role,
        "content": content,
    }
    message.update(extras)
    return message


# Tests for get_user_message
def test_get_user_message_single_user() -> None:
    """Test extracting a single user message from trace."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message("user", "Hello"),
        _make_message("assistant", "Hi there!"),
    ]
    
    result, idxs = get_user_message(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"
    assert len(idxs) == 1
    assert idxs[0] == 1


def test_get_user_message_no_user() -> None:
    """Test when no user message exists."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message("assistant", "Hi there!"),
    ]
    
    result, idxs = get_user_message(messages)
    assert result == []
    assert idxs == []


def test_get_user_message_empty() -> None:
    """Test with empty messages list."""
    result, idxs = get_user_message([])
    assert result == []
    assert idxs == []


def test_get_user_message_multiple_users() -> None:
    """Test extracting all user messages from trace."""
    messages = [
        _make_message("user", "First question"),
        _make_message("assistant", "First answer"),
        _make_message("user", "Second question"),
        _make_message("assistant", "Second answer"),
    ]
    
    result, idxs = get_user_message(messages)
    assert len(result) == 2
    assert result[0]["content"] == "First question"
    assert result[1]["content"] == "Second question"
    assert len(idxs) == 2


# Tests for get_last_tool_interaction
def test_get_last_tool_interaction_valid() -> None:
    """Test extracting a valid tool episode."""
    messages = [
        _make_message("user", "Get data"),
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[{"id": "tc-1", "type": "function", "function": {"name": "fetch"}}],
        ),
        _make_message("tool", "Result 1", tool_call_id="tc-1"),
    ]
    
    interaction, idx = get_last_tool_interaction(messages)
    assert len(interaction) == 2  # assistant + 1 tool response
    assert interaction[0]["role"] == "assistant"
    assert "tool_calls" in interaction[0]
    assert interaction[1]["role"] == "tool"
    assert idx == 1


def test_get_last_tool_interaction_no_tools() -> None:
    """Test when no tool messages exist at end."""
    messages = [
        _make_message("user", "Hello"),
        _make_message("assistant", "Hi there!"),
    ]
    
    interaction, idx = get_last_tool_interaction(messages)
    assert interaction == []
    assert idx == len(messages)


def test_get_last_tool_interaction_empty() -> None:
    """Test with empty messages list."""
    interaction, idx = get_last_tool_interaction([])
    assert interaction == []
    assert idx == 0


def test_get_last_tool_interaction_multiple_parallel() -> None:
    """Test multiple parallel tool calls from single assistant message."""
    messages = [
        _make_message("user", "Get multiple data"),
        _make_message(
            "assistant",
            "Fetching multiple...",
            tool_calls=[
                {"id": "tc-1", "type": "function", "function": {"name": "fetch1"}},
                {"id": "tc-2", "type": "function", "function": {"name": "fetch2"}},
            ],
        ),
        _make_message("tool", "Result 1", tool_call_id="tc-1"),
        _make_message("tool", "Result 2", tool_call_id="tc-2"),
    ]
    
    interaction, idx = get_last_tool_interaction(messages)
    assert len(interaction) == 3  # 1 assistant + 2 tool responses
    assert interaction[0]["role"] == "assistant"
    assert len(interaction[0]["tool_calls"]) == 2
    assert interaction[1]["role"] == "tool"
    assert interaction[2]["role"] == "tool"
    assert idx == 1


def test_get_last_tool_interaction_tool_id_mismatch() -> None:
    """Test when tool response IDs don't match assistant's tool_call IDs."""
    # Structure-based parsing accepts this
    messages = [
        _make_message("user", "Get data"),
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[{"id": "tc-1", "type": "function", "function": {"name": "fetch"}}],
        ),
        _make_message("tool", "Result", tool_call_id="tc-999"),  # Mismatched ID
    ]
    
    interaction, idx = get_last_tool_interaction(messages)
    assert len(interaction) == 2
    assert interaction[0]["role"] == "assistant"


def test_get_last_tool_interaction_no_preceding_assistant() -> None:
    """Test when tool messages have no preceding assistant message."""
    messages = [
        _make_message("user", "Get data"),
        _make_message("tool", "Orphan tool result", tool_call_id="tc-1"),
    ]
    
    interaction, idx = get_last_tool_interaction(messages)
    assert interaction == []
    assert idx == len(messages)



# Tests for get_first_user_text
def test_get_first_user_text_returns_first_user_content() -> None:
    """get_first_user_text should return the content of the first user message when
    multiple user messages are present, ignoring any subsequent ones.
    """
    messages = [
        _make_message("system", "System prompt"),
        _make_message("user", "First question"),
        _make_message("assistant", "Answer"),
        _make_message("user", "Second question"),
    ]
    assert get_first_user_text(messages) == "First question"


def test_get_first_user_text_no_user_returns_empty_string() -> None:
    """get_first_user_text should return an empty string when there are no user
    messages, so callers can treat the result as a plain string without None checks.
    """
    messages = [
        _make_message("system", "System prompt"),
        _make_message("assistant", "Hello"),
    ]
    assert get_first_user_text(messages) == ""


def test_get_first_user_text_empty_messages_returns_empty_string() -> None:
    """get_first_user_text should handle an empty message list gracefully and return ''."""
    assert get_first_user_text([]) == ""


def test_get_first_user_text_single_user() -> None:
    """get_first_user_text should return the content when exactly one user message exists."""
    messages = [_make_message("user", "Hello")]
    assert get_first_user_text(messages) == "Hello"


# Tests for extract_tool_outputs
def test_extract_tool_outputs_single_call() -> None:
    """extract_tool_outputs should return a single (tool_name, raw_input, raw_output)
    tuple when there is exactly one tool call and its corresponding tool response.
    """
    import json
    messages = [
        _make_message("user", "fetch something"),
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[{
                "id": "tc-1",
                "type": "function",
                "function": {"name": "fetch_data", "arguments": json.dumps({"url": "http://example.com"})},
            }],
        ),
        _make_message("tool", json.dumps({"result": "data"}), tool_call_id="tc-1"),
    ]

    outputs = extract_tool_outputs(messages)

    assert len(outputs) == 1
    tool_name, raw_input, raw_output = outputs[0]
    assert tool_name == "fetch_data"
    assert raw_input == {"url": "http://example.com"}
    assert raw_output == {"result": "data"}


def test_extract_tool_outputs_multiple_parallel_calls() -> None:
    """extract_tool_outputs should handle multiple parallel tool calls from a single
    assistant message, returning one tuple per tool call in encounter order.
    """
    import json
    messages = [
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[
                {"id": "tc-1", "type": "function", "function": {"name": "tool_a", "arguments": json.dumps({"x": 1})}},
                {"id": "tc-2", "type": "function", "function": {"name": "tool_b", "arguments": json.dumps({"y": 2})}},
            ],
        ),
        _make_message("tool", json.dumps({"a": "result"}), tool_call_id="tc-1"),
        _make_message("tool", json.dumps({"b": "result"}), tool_call_id="tc-2"),
    ]

    outputs = extract_tool_outputs(messages)

    assert len(outputs) == 2
    assert outputs[0][0] == "tool_a"
    assert outputs[0][1] == {"x": 1}
    assert outputs[1][0] == "tool_b"
    assert outputs[1][1] == {"y": 2}


def test_extract_tool_outputs_no_tool_calls_returns_empty() -> None:
    """extract_tool_outputs should return an empty list when no assistant message
    with tool_calls is present in the trace.
    """
    messages = [
        _make_message("user", "Hello"),
        _make_message("assistant", "Hi"),
    ]
    assert extract_tool_outputs(messages) == []


def test_extract_tool_outputs_empty_messages_returns_empty() -> None:
    """extract_tool_outputs should handle an empty message list by returning []."""
    assert extract_tool_outputs([]) == []


def test_extract_tool_outputs_non_json_tool_response() -> None:
    """extract_tool_outputs should handle tool responses that are not valid JSON,
    storing them under the '_raw' key in the raw_output dict.
    """
    import json
    messages = [
        _make_message(
            "assistant",
            "Running...",
            tool_calls=[{
                "id": "tc-1",
                "type": "function",
                "function": {"name": "run_cmd", "arguments": json.dumps({"cmd": "ls"})},
            }],
        ),
        _make_message("tool", "not json at all", tool_call_id="tc-1"),
    ]

    outputs = extract_tool_outputs(messages)

    assert len(outputs) == 1
    _, _, raw_output = outputs[0]
    assert raw_output == {"_raw": "not json at all"}
