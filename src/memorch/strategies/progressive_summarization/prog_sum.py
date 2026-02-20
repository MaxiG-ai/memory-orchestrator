from typing import Dict, List, Optional
from pathlib import Path

from memorch.utils.llm_helpers import extract_content
from memorch.utils.logger import get_logger
from memorch.utils.split_trace import process_and_split_trace_user

logger = get_logger("ProgressiveSummarization")


def _resolve_prompt_path(prompt_path: Optional[str]) -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    if prompt_path:
        candidate = Path(prompt_path)
        if candidate.is_file():
            return candidate
        candidate_from_root = repo_root / prompt_path
        if candidate_from_root.is_file():
            return candidate_from_root
        candidate_from_site_packages = repo_root / ".venv" / "lib" / "python3.13" / "site-packages" / "memorch" / "strategies" / "progressive_summarization" / "prog_sum.prompt.md"
        if candidate_from_site_packages.is_file():
            return candidate_from_site_packages
    return repo_root / "src/memorch/strategies/progressive_summarization/prog_sum.prompt.md"


def summarize_conv_history(
    messages: List[Dict],
    llm_client,
    summarizer_model: str = "gpt-4-1",
    summary_prompt_path: Optional[str] = None,
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

    user_query, conversation_history = process_and_split_trace_user(messages)

    prompt_file = _resolve_prompt_path(summary_prompt_path)
    summarization_prompt = prompt_file.read_text(encoding="utf-8")

    # Build prompt for summarization
    prompt_messages = [
        {"role": "system", "content": summarization_prompt},
        {
            "role": "user",
            "content": f"Conversation history to compress:\n{conversation_history}",
        },
    ]

    # Call LLM to generate summary
    response = llm_client.generate_plain(
        input_messages=prompt_messages, 
        model=summarizer_model
    )
    summary_text = extract_content(response)

    if not summary_text:
        raise ValueError("Summarization returned empty content")

    # Build final message list: [summary, user query]
    summary_message = {"role": "system", "content": summary_text}

    result = []
    if user_query:
        result.extend(user_query)
    result.extend([summary_message])

    return result
