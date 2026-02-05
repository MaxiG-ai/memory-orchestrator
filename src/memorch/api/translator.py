"""
Translation layer between OpenAI Chat-Completions and Open Responses formats.

This module provides bidirectional conversion between:
- OpenAI chat-completions format: Used internally by LLMOrchestrator
  (messages with role/content/tool_calls)
- Open Responses format: Used by the external API
  (items with type-specific structures)

The translator handles:
- User/system messages <-> MessageItem with InputTextContent
- Assistant messages <-> MessageItem with OutputTextContent
- tool_calls in assistant messages <-> separate FunctionCallItem entries
- Tool role messages <-> FunctionCallOutputItem

Functions:
    chat_completions_to_items: Convert chat-completions messages to Open Responses items
    items_to_chat_completions: Convert Open Responses items to chat-completions messages
"""

from typing import List, Dict, Union, Any
from memorch.api.models import (
    Item,
    MessageItem,
    FunctionCallItem,
    FunctionCallOutputItem,
    InputTextContent,
    OutputTextContent,
    MessageRole,
    ItemStatus,
)


def chat_completions_to_items(messages: List[Dict[str, Any]]) -> List[Item]:
    """
    Convert OpenAI chat-completions messages to Open Responses items.

    Transforms the chat-completions format (flat list of messages with role/content)
    into the Open Responses format (typed items with structured content).

    Args:
        messages: List of chat-completions messages, each with:
            - role: "user", "assistant", "system", or "tool"
            - content: String message content (can be None)
            - tool_calls: Optional list of tool calls (for assistant messages)
            - tool_call_id: ID linking to original call (for tool messages)

    Returns:
        List of Open Responses items (MessageItem, FunctionCallItem, or
        FunctionCallOutputItem) preserving conversation order.

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi!",
        ...      "tool_calls": [{"id": "c1", "function": {"name": "fn", "arguments": "{}"}}]},
        ...     {"role": "tool", "tool_call_id": "c1", "content": "result"}
        ... ]
        >>> items = chat_completions_to_items(messages)
        >>> len(items)  # user + assistant + function_call + function_output
        4
    """
    items: List[Item] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        # Handle tool responses (role="tool")
        if role == "tool":
            items.append(
                FunctionCallOutputItem(
                    call_id=msg.get("tool_call_id", ""),
                    output=content or "",
                    status=ItemStatus.completed,
                )
            )
            continue

        # Build content list based on role
        content_list = _build_content_list(role, content)

        # Create MessageItem for user/assistant/system
        try:
            msg_role = MessageRole(role)
        except ValueError:
            # Skip unknown roles
            continue

        items.append(
            MessageItem(
                role=msg_role,
                content=content_list,
                status=ItemStatus.completed,
            )
        )

        # Extract tool_calls from assistant messages into FunctionCallItems
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls or []:
            func = tc.get("function", {})
            items.append(
                FunctionCallItem(
                    call_id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=func.get("arguments", "{}"),
                    status=ItemStatus.completed,
                )
            )

    return items


def _build_content_list(
    role: str, content: Any
) -> List[Union[InputTextContent, OutputTextContent]]:
    """
    Build the appropriate content list for a message role.

    Args:
        role: Message role ("user", "assistant", "system")
        content: Raw content (string, None, or empty)

    Returns:
        List of content objects (InputTextContent or OutputTextContent).
        Empty list if content is None or empty string.
    """
    # Handle empty/None content
    if not content:
        return []

    # Assistant messages use OutputTextContent, others use InputTextContent
    if role == "assistant":
        return [OutputTextContent(text=str(content))]
    return [InputTextContent(text=str(content))]


def items_to_chat_completions(
    items: List[Union[Item, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Convert Open Responses items to OpenAI chat-completions messages.

    Transforms Open Responses format (typed items) back into chat-completions
    format (flat messages with role/content). FunctionCallItems are grouped
    into the preceding assistant message's tool_calls array.

    Args:
        items: List of Open Responses items. Can be Pydantic models or dicts.
            Supported types: MessageItem, FunctionCallItem, FunctionCallOutputItem

    Returns:
        List of chat-completions messages, each with:
            - role: "user", "assistant", "system", or "tool"
            - content: String or None
            - tool_calls: List of tool calls (for assistant messages, if any)
            - tool_call_id: Call ID (for tool messages)

    Note:
        FunctionCallItems are attached to the preceding assistant message.
        If no preceding assistant message exists, one is created with content=None.

    Example:
        >>> items = [
        ...     MessageItem(role=MessageRole.user, content=[InputTextContent(text="Hi")]),
        ...     MessageItem(role=MessageRole.assistant, content=[OutputTextContent(text="Hello")]),
        ...     FunctionCallItem(call_id="c1", name="fn", arguments="{}", status=ItemStatus.completed),
        ... ]
        >>> messages = items_to_chat_completions(items)
        >>> messages[1]["tool_calls"][0]["id"]
        'c1'
    """
    messages: List[Dict[str, Any]] = []

    for item in items:
        item_type = _get_item_type(item)

        if item_type == "message":
            messages.append(_convert_message_item(item))

        elif item_type == "function_call":
            _attach_function_call(messages, item)

        elif item_type == "function_call_output":
            messages.append(_convert_function_output(item))

    return messages


def _get_item_type(item: Union[Item, Dict[str, Any]]) -> str:
    """Get the type of an item (works with both Pydantic models and dicts)."""
    if hasattr(item, "type"):
        return item.type
    if isinstance(item, dict):
        return item.get("type", "")
    return ""


def _get_attr(item: Union[Any, Dict], attr: str, default: Any = None) -> Any:
    """Get an attribute from an item (works with both Pydantic models and dicts)."""
    if isinstance(item, dict):
        return item.get(attr, default)
    return getattr(item, attr, default)


def _convert_message_item(item: Union[MessageItem, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a MessageItem to a chat-completions message dict.

    Handles both Pydantic MessageItem and dict representations.
    Concatenates multiple content items with newlines.
    """
    role = _get_attr(item, "role")
    # Handle both enum and string roles
    if hasattr(role, "value"):
        role = role.value

    content = _get_attr(item, "content", [])
    content_str = _extract_content_string(content)

    return {
        "role": role,
        "content": content_str,
    }


def _extract_content_string(content: Any) -> Union[str, None]:
    """
    Extract a content string from various content formats.

    Handles:
    - String content (passthrough)
    - List of content objects (concatenate text fields)
    - Empty list/None (returns None)
    """
    # String content (shorthand format)
    if isinstance(content, str):
        return content

    # Empty or None content
    if not content:
        return None

    # List of content objects
    texts = []
    for c in content:
        text = _get_attr(c, "text", "")
        if text:
            texts.append(text)

    return "\n".join(texts) if texts else None


def _attach_function_call(
    messages: List[Dict[str, Any]], item: Union[FunctionCallItem, Dict[str, Any]]
) -> None:
    """
    Attach a FunctionCallItem to the appropriate assistant message.

    If the last message is an assistant message, appends to its tool_calls.
    Otherwise, creates a new assistant message with content=None.
    """
    tool_call = {
        "id": _get_attr(item, "call_id"),
        "type": "function",
        "function": {
            "name": _get_attr(item, "name"),
            "arguments": _get_attr(item, "arguments"),
        },
    }

    # Try to attach to last assistant message
    if messages and messages[-1].get("role") == "assistant":
        if "tool_calls" not in messages[-1]:
            messages[-1]["tool_calls"] = []
        messages[-1]["tool_calls"].append(tool_call)
    else:
        # Create new assistant message for the tool calls
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
        )


def _convert_function_output(
    item: Union[FunctionCallOutputItem, Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert a FunctionCallOutputItem to a tool role message."""
    return {
        "role": "tool",
        "tool_call_id": _get_attr(item, "call_id"),
        "content": _get_attr(item, "output"),
    }
