"""
Reflector agent for ACE strategy.
Analyzes performance and tags playbook bullets.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from memorch.strategies.ace.playbook_utils import extract_json_from_text
from memorch.utils.llm_helpers import extract_content
from memorch.utils.prompt_manager import PromptManager
from memorch.utils.logger import get_logger

logger = get_logger("ACE.Reflector")

_PROMPT_FILE = "reflector.prompt.md"
_PROMPTS_DIR = Path(__file__).parent / "prompts"


class Reflector:
    """
    Reflector agent that evaluates performance and tags bullets.
    """

    def __init__(self, prompt_path: str | None = None):
        """
        Initialize reflector with prompt template via PromptManager.

        Args:
            prompt_path: Path to prompt file (defaults to reflector.prompt.md
                         bundled alongside this module).
        """
        self._prompt_manager = PromptManager(
            prompt_file_name=_PROMPT_FILE,
            prompt_path=prompt_path or "",
            caller_dir=_PROMPTS_DIR,
        )

    def reflect(
        self,
        question: str,
        reasoning_trace: str,
        predicted_answer: str,
        environment_feedback: str,
        bullets_used: str,
        llm_client,
        model: str = "gpt-4-1-mini",
    ) -> Tuple[str, List[Dict]]:
        """
        Reflect on performance and tag bullets.

        Args:
            question: The question/task
            reasoning_trace: Agent's reasoning process
            predicted_answer: Agent's answer
            environment_feedback: Feedback from environment
            bullets_used: Formatted bullets that were used
            llm_client: LLM client
            model: Model to use

        Returns:
            (reflection_text, bullet_tags)
            bullet_tags: List of dicts with bullet_id and tag
        """
        # Build prompt via PromptManager (uses $-style Template substitution)
        prompt = self._prompt_manager.render(
            question=question,
            reasoning_trace=reasoning_trace,
            predicted_answer=predicted_answer,
            environment_feedback=environment_feedback or "No feedback",
            bullets_used=bullets_used or "No bullets used",
        )

        logger.debug(
            f"Reflector input - reasoning_trace (first 200 chars): {reasoning_trace[:200] if reasoning_trace else '<empty>'}..."
        )

        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = llm_client.generate_plain(input_messages=messages, model=model)
        response_text = extract_content(response)

        # Extract bullet tags
        bullet_tags = self._extract_bullet_tags(response_text)
        logger.debug(f"Reflector extracted bullet_tags: {bullet_tags}")

        return response_text, bullet_tags

    def _extract_bullet_tags(self, text: str) -> List[Dict]:
        """
        Extract bullet tags from reflection response.

        Args:
            text: Response text

        Returns:
            List of dicts with bullet_id and tag
        """
        # Try JSON extraction
        json_data = extract_json_from_text(text)
        if json_data and "bullet_tags" in json_data:
            tags = json_data["bullet_tags"]
            if isinstance(tags, list):
                try:
                    return [
                        {
                            "bullet_id": int(tag["bullet_id"]),
                            "tag": tag["tag"],
                        }
                        for tag in tags
                    ]
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(
                        f"⚠ Error parsing bullet_tags from JSON: {e}. Falling back to text extraction."
                    )

        return []
