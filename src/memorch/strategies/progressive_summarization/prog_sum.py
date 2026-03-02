from typing import Dict, List

from memorch.utils.llm_helpers import extract_content
from memorch.utils.prompt_manager import PromptManager
from memorch.utils.logger import get_logger
from memorch.utils.split_trace import get_first_user_text

logger = get_logger("ProgressiveSummarization")


def summarize_conv_history(
    messages: List[Dict],
    llm_client,
    summary_prompt_path: str,
    summarizer_model: str = "gpt-4-1",
) -> List[Dict]:
    """
    Splitting a message trace into user query and conversation history, then summarizing the conversation history using an LLM,
    and finally returning a new message list that includes the summary and the user query.

    :param messages: Description
    :type messages: List[Dict]
    :param llm_client: Description
    :param summarizer_model: Description
    :type summarizer_model: str
    :param summary_prompt_path: Description
    :type summary_prompt_path: Optional[str]
    :return: Description
    :rtype: List[Dict[Any, Any]]
    """
    if llm_client is None:
        raise ValueError("llm_client is required for progressive summarization")

    # TODO: This should be changed to not depend on the order of messages
    user_query_text = get_first_user_text(messages)
    user_query = {"role": "user", "content": user_query_text} if user_query_text else None

    prog_sum_prompt = PromptManager(
        prompt_file_name="prog_sum.prompt.md", prompt_path=summary_prompt_path
    )
    summarization_prompt = prog_sum_prompt.render(user_query=user_query)

    prompt_messages = [
        {"role": "system", "content": summarization_prompt},
        {
            "role": "user",
            "content": f"Conversation history to compress:\n{messages}",
        },
    ]

    # Call LLM to generate summary
    response = llm_client.generate_plain(
        input_messages=prompt_messages, model=summarizer_model
    )
    summary_text = extract_content(response)

    if not summary_text:
        raise ValueError("Summarization returned empty content")

    # Build final message list: [user query, summary]
    summary_message = {"role": "system", "content": summary_text}

    result = []
    if user_query:
        result.append(user_query)  # append the dict, not extend over its keys
    result.append(summary_message)

    return result
