"""
Tests for prompt_manager._resolve_prompt_path.

Covers the three-step resolution order introduced to fix path resolution when
memorch is installed as an external package:
  1. ``prompt_path`` argument  – used when it points to an existing file.
  2. ``caller_dir``            – used when provided and ``prompt_path`` is absent
                                 or does not resolve to a file.
  3. ``Path(__file__).parent`` – last-resort legacy fallback (utils/ directory).
"""

from pathlib import Path

import pytest

from memorch.utils.prompt_manager import PromptManager, _resolve_prompt_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prompt_file(tmp_path) -> Path:
    """Create a real prompt file in a temporary directory."""
    p = tmp_path / "test.prompt.md"
    p.write_text("Hello ${name}")
    return p


@pytest.fixture
def caller_dir_with_prompt(tmp_path) -> Path:
    """
    Create a temporary directory that contains a prompt file named
    'test.prompt.md', simulating a bundled prompt sitting next to its module.
    """
    (tmp_path / "test.prompt.md").write_text("From caller dir: ${name}")
    return tmp_path


# ---------------------------------------------------------------------------
# _resolve_prompt_path
# ---------------------------------------------------------------------------


def test_explicit_prompt_path_takes_priority(prompt_file, tmp_path):
    """
    When ``prompt_path`` points to an existing file it must be returned
    regardless of whether ``caller_dir`` is also provided.

    This ensures that users who configure a custom path keep their override.
    """
    caller_dir = tmp_path / "other"
    caller_dir.mkdir()
    (caller_dir / "test.prompt.md").write_text("Should not be used")

    result = _resolve_prompt_path("test.prompt.md", str(prompt_file), caller_dir)

    assert result == prompt_file


def test_falls_back_to_caller_dir_when_prompt_path_missing(caller_dir_with_prompt):
    """
    When ``prompt_path`` is an empty string (the default config value for a
    path that isn't set) the function must fall back to
    ``caller_dir / prompt_file_name``.

    This is the core of the bug-fix: installed packages cannot rely on a
    relative ``prompt_path`` resolving against the CWD of the consuming project.
    """
    result = _resolve_prompt_path("test.prompt.md", "", caller_dir_with_prompt)

    assert result == caller_dir_with_prompt / "test.prompt.md"


def test_falls_back_to_caller_dir_when_prompt_path_nonexistent(caller_dir_with_prompt, tmp_path):
    """
    When ``prompt_path`` is a relative path that does not exist as a file
    (e.g. the default config value ``'memorch/strategies/.../prog_sum.prompt.md'``
    evaluated from a different working directory), the function must fall back
    to ``caller_dir / prompt_file_name``.
    """
    nonexistent = str(tmp_path / "does" / "not" / "exist.prompt.md")

    result = _resolve_prompt_path("test.prompt.md", nonexistent, caller_dir_with_prompt)

    assert result == caller_dir_with_prompt / "test.prompt.md"


def test_falls_back_to_utils_dir_without_caller_dir(tmp_path):
    """
    When neither a valid ``prompt_path`` nor a ``caller_dir`` is supplied,
    the function falls back to ``Path(__file__).parent`` (the ``utils/``
    directory) for backward-compatibility with callers that bundle their
    prompt next to ``prompt_manager.py``.
    """
    result = _resolve_prompt_path("any.prompt.md", "", caller_dir=None)

    expected = Path(__file__).parent.parent.parent / "src" / "memorch" / "utils" / "any.prompt.md"
    # The resolved path should live inside the utils/ package directory.
    assert result.parent.name == "utils"
    assert result.name == "any.prompt.md"


# ---------------------------------------------------------------------------
# PromptManager integration
# ---------------------------------------------------------------------------


def test_prompt_manager_reads_file_via_caller_dir(caller_dir_with_prompt):
    """
    PromptManager must successfully load a prompt when ``caller_dir`` is given
    and ``prompt_path`` does not resolve to an existing file.

    This reproduces the production scenario: the default ``summary_prompt``
    config value is a relative path such as
    ``'memorch/strategies/progressive_summarization/prog_sum.prompt.md'``
    which does not exist when the package is installed elsewhere.
    With ``caller_dir=Path(__file__).parent`` passed from ``prog_sum.py``,
    the bundled prompt file is found correctly.
    """
    pm = PromptManager(
        prompt_file_name="test.prompt.md",
        prompt_path="memorch/strategies/progressive_summarization/prog_sum.prompt.md",  # won't exist
        caller_dir=caller_dir_with_prompt,
    )

    assert "From caller dir:" in pm.template


def test_prompt_manager_render_with_caller_dir(caller_dir_with_prompt):
    """
    End-to-end: PromptManager loaded via ``caller_dir`` renders variables
    correctly with ``Template.safe_substitute``.
    """
    pm = PromptManager(
        prompt_file_name="test.prompt.md",
        prompt_path="",
        caller_dir=caller_dir_with_prompt,
    )

    rendered = pm.render(name="world")

    assert rendered == "From caller dir: world"


def test_prog_sum_uses_bundled_prompt_without_valid_prompt_path():
    """
    Verify that the actual ``prog_sum.prompt.md`` bundled with the package is
    found when ``summary_prompt_path`` is the default config relative path.

    This is the exact failure scenario reported: when memorch is installed as
    an external package the default ``'memorch/strategies/.../prog_sum.prompt.md'``
    path does not resolve from the consuming project's CWD. The fix in
    ``prog_sum.py`` passes ``caller_dir=Path(__file__).parent`` so the bundled
    file is always located relative to the installed module.
    """
    from memorch.strategies.progressive_summarization import prog_sum as ps_module

    # Simulate the default (broken) relative config path
    default_config_path = "memorch/strategies/progressive_summarization/prog_sum.prompt.md"

    pm = PromptManager(
        prompt_file_name="prog_sum.prompt.md",
        prompt_path=default_config_path,
        caller_dir=Path(ps_module.__file__).parent,
    )

    assert pm.template  # non-empty → file was found
