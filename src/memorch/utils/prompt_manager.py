import json
from string import Template
from pathlib import Path
from typing import Optional

def _resolve_prompt_path(
    prompt_file_name: str,
    prompt_path: str,
    caller_dir: Optional[Path] = None,
) -> Path:
    """Resolve path to a prompt file.

    Resolution order:
    1. ``prompt_path`` if it points to an existing file.
    2. ``caller_dir / prompt_file_name`` if ``caller_dir`` is given — allows
       installed packages to locate bundled prompts via ``Path(__file__).parent``
       in the calling module rather than this utility module.
    3. ``Path(__file__).parent / prompt_file_name`` as a last-resort fallback.
    """
    if prompt_path:
        candidate = Path(prompt_path)
        if candidate.is_file():
            return candidate
    # Use the caller's directory so bundled prompt files are found after
    # installation, regardless of the current working directory.
    base = caller_dir if caller_dir is not None else Path(__file__).parent
    return base / prompt_file_name

class PromptManager:
    def __init__(self, prompt_file_name: str, prompt_path: str, caller_dir: Optional[Path] = None):
        # Store raw template strings
        self.template = _resolve_prompt_path(prompt_file_name, prompt_path, caller_dir).read_text(encoding="utf-8")
        assert self.template, f"Prompt template at {prompt_file_name} is empty or not found."
        assert type(self.template) is str, f"Prompt template at {prompt_file_name} must be a string."

    def render(self, **kwargs) -> str:
        """
        Renders a template. Uses safe_substitute so missing variables 
        leave the $placeholder intact instead of throwing a KeyError.
        """
        prompt = Template(self.template)
        return prompt.safe_substitute(**kwargs)

    def render_json_prompt(self, name: str, **kwargs) -> str:
        """Helper to ensure injected dictionaries are properly JSON formatted for the LLM."""
        formatted_kwargs = {
            k: (json.dumps(v, indent=2) if isinstance(v, (dict, list)) else v)
            for k, v in kwargs.items()
        }
        return self.render(**formatted_kwargs)