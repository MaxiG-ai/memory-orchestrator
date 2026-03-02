from typing import List, Dict
from memorch.utils.logger import get_logger
from memorch.utils.split_trace import get_user_message, get_last_tool_interaction
from memorch.utils.token_count import get_token_count

logger = get_logger("TruncationStrategy")

def truncate_messages(
    messages: List[Dict],
) -> List[Dict]:
    """ Truncation strategy that splits messages into user query, conversation history, and tool interaction.
        Keeps the last user query and the last tool interaction intact,
        truncating the conversation history in between as needed.
    """

    user_msgs, _ = get_user_message(messages)
    user_query = user_msgs[:1]  # keep only the first user message (the task)
    tool_interaction, _ = get_last_tool_interaction(messages)
    logger.debug(f"""🧠 Truncation Strategy: 
                User Query Tokens: {get_token_count(user_query)},
                Tool Interaction Tokens: {get_token_count(tool_interaction)}"""
                 )
    return user_query + tool_interaction