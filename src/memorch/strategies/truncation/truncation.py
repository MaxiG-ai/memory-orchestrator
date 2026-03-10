from typing import List, Dict
from memorch.utils.logger import get_logger
from memorch.utils.split_trace import get_user_message, get_last_tool_interaction

logger = get_logger("TruncationStrategy")

def truncate_messages(
    messages: List[Dict],
    keep_last_n_tool_interactions: int = 1,
) -> List[Dict]:
    """ Truncation strategy that splits messages into user query, conversation history, and tool interaction.
        Keeps the first user query and the last n tool interaction intact,
        truncating the conversation history in between as needed.
    """
    user_msgs, _ = get_user_message(messages)
    user_query = user_msgs[:1]  # keep only the first user message (the task)
    if len(user_msgs) > 1:
        logger.warning("🚨 More than 1 user message found in trace, keeping only the first as the user query.")

    old_messages = messages.copy()
    return_messages = user_query
    keep_count = 0
    while keep_count < keep_last_n_tool_interactions and old_messages:
        tool_interaction, _ = get_last_tool_interaction(old_messages)
        return_messages += tool_interaction
        old_messages = old_messages[:-len(tool_interaction)]  # Remove processed tool interaction from old_messages
        keep_count += 1
    return return_messages