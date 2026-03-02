# contains utilities to split llm traces
from typing import List, Dict, Tuple
import json


def get_user_message(messages: List[Dict]) -> Tuple[List[Dict], List[int]]:
    """Get user message(s) from a list of messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Tuple of (user_messages, user_message_indices) where user_messages is a list of user message dicts and user_message_indices is a list of their indices.
    """
    if not messages:
        return [], []
    
    user_messages = []
    user_messages_idx = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            user_messages.append(msg)
            user_messages_idx.append(i)
    
    return user_messages, user_messages_idx 


def get_last_tool_interaction(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """Get the last valid tool interaction from a list of messages.
    
    Iterates through messages to find tool interactions following the pattern:
    Assistant (with tool_calls) -> One or more Tool messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Tuple of (tool_episode, start_index) where:
        - tool_episode: List containing [assistant_msg, tool_msg1, ...], or empty list
        - start_index: Index where the tool episode starts, or len(messages) if not found
    """
    if not messages:
        return [], len(messages)

    last_interaction = ([], len(messages))
    i = 0
    
    while i < len(messages):
        msg = messages[i]
        
        # Check for assistant message with tool_calls
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            current_interaction = [msg]
            start_idx = i
            
            # Look ahead for tool messages
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                current_interaction.append(messages[j])
                j += 1
            
            # If we found at least one tool response, this is a valid interaction
            if len(current_interaction) > 1:
                last_interaction = (current_interaction, start_idx)
            
            # Advance main pointer
            i = j
        else:
            i += 1
            
    return last_interaction


def process_and_split_trace_user(messages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split a trace into the last user message and everything *after* it.

    Note:
        In ComplexFuncBench traces there is typically a single initial user message
        followed by system/assistant/tool traffic. This helper is used to isolate
        that user message and return the remainder of the trace.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Tuple of ([last_user_message], messages_after_last_user)
        If no user message found, returns ([], all_messages)
    """
    if not messages:
        return [], []
    
    user_messages, user_message_idx = get_user_message(messages)
    
    if not user_messages:
        return [], messages

    conv_after_user_message = messages[user_message_idx[-1] + 1 :]
    
    return [user_messages[-1]], conv_after_user_message

# Use by prog_sum
def process_full_trace_split_user(messages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Process a full trace and split into user messages and non-user messages.
    
    This is a simpler split that just separates out all user messages from the rest of the trace, without assuming any order.
    
    Args:
        messages: List of message dictionaries
    Returns:
        Tuple of (user_messages, non_user_messages) where:
        - user_messages: List of all messages with role "user"
        - non_user_messages: List of all messages that are not from the user
    """
    assert messages is not None, "Messages list cannot be None"
    
    user_messages = []
    non_user_messages = []
    
    for msg in messages:
        if msg.get("role") == "user":
            user_messages.append(msg)
        else:
            non_user_messages.append(msg)
    
    return user_messages, non_user_messages


def process_and_split_trace_user_tool(messages: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Process and split the trace into first user message, intermediate messages, and last tool episode.
    
    This function performs a 3-way split of the conversation:
    - First user message (the initial query)
    - Intermediate messages (between first user and the tool episode)
    - Last tool episode (assistant with tool_calls + tool responses)
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Tuple of ([first_user_message], intermediate_messages, last_tool_episode)
        where:
        - first_user_message: List with the first user message (empty if not found)
        - intermediate_messages: All messages between first user and tool episode
        - last_tool_episode: The last valid tool episode (empty if not found)
    """
    if not messages:
        return [], [], []
    
    # Get all user messages and use the first one
    user_messages, user_messages_idx = get_user_message(messages)
    
    # Extract the last valid tool episode and its start index
    last_tool_episode, tool_episode_start_idx = get_last_tool_interaction(messages)
    
    if not user_messages:
        return [], messages[:tool_episode_start_idx], last_tool_episode

    # Intermediate messages are between first user and the tool episode start
    intermediate_start = user_messages_idx[0] + 1
    intermediate_end = tool_episode_start_idx if last_tool_episode else len(messages)
    intermediate_messages = messages[intermediate_start:intermediate_end]
    
    return [user_messages[0]], intermediate_messages, last_tool_episode


def extract_tool_outputs(messages: List[Dict]) -> List[Tuple[str, Dict, Dict]]:
    """
    Extract tool call information from the latest tool interaction in messages.

    Parses assistant messages with tool_calls and subsequent tool responses
    to extract (tool_name, raw_input, raw_output) tuples.

    Args:
        messages: Conversation messages list

    Returns:
        List of (tool_name, raw_input, raw_output) tuples from the last
        tool interaction. Empty list if no tool calls found.
    """
    if not messages:
        return []

    # Find the last assistant message with tool_calls
    assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            assistant_idx = i
            break

    if assistant_idx is None:
        return []

    assistant_msg = messages[assistant_idx]
    tool_calls = assistant_msg.get("tool_calls", [])

    # Build mapping of tool_call_id -> (tool_name, raw_input)
    call_map: Dict[str, Tuple[str, Dict]] = {}
    for tc in tool_calls:
        # Handle both dict and object formats
        if isinstance(tc, dict):
            call_id = tc.get("id", "")
            func = tc.get("function", {})
            tool_name = func.get("name", "unknown")
            args_str = func.get("arguments", "{}")
        else:
            call_id = getattr(tc, "id", "")
            func = getattr(tc, "function", None)
            tool_name = getattr(func, "name", "unknown") if func else "unknown"
            args_str = getattr(func, "arguments", "{}") if func else "{}"

        # Parse arguments JSON
        try:
            raw_input = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            raw_input = {"_raw": args_str}

        call_map[call_id] = (tool_name, raw_input)

    # Collect tool responses that follow the assistant message
    outputs = []
    for msg in messages[assistant_idx + 1 :]:
        if msg.get("role") != "tool":
            break

        tool_call_id = msg.get("tool_call_id", "")
        content_str = msg.get("content", "{}")

        # Parse response JSON
        try:
            raw_output = json.loads(content_str) if content_str else {}
        except json.JSONDecodeError:
            raw_output = {"_raw": content_str}

        if tool_call_id in call_map:
            tool_name, raw_input = call_map[tool_call_id]
            outputs.append((tool_name, raw_input, raw_output))

    return outputs