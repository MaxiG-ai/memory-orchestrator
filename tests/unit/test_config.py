import textwrap

import pytest

from memorch.utils.config import MemoryDef, load_configs


def test_memory_def_accepts_auto_compact_threshold():
    definition = MemoryDef(
        type="progressive_summarization",
        auto_compact_threshold=4000,
        summarizer_model="gpt-5-mini",
    )

    assert definition.auto_compact_threshold == 4000
    assert definition.summarizer_model == "gpt-5-mini"


# ---------------------------------------------------------------------------
# Minimal TOML helpers
# ---------------------------------------------------------------------------

_MINIMAL_EXPERIMENT_CONFIG = textwrap.dedent("""\
    experiment_name = "test"
    results_dir = "results"
    log_dir = "logs"
    logging_level = "WARNING"
    input_file = "data.jsonl"
    enabled_models = ["gpt-test"]
    enabled_memory_methods = ["no_strategy"]
    compact_thresholds = [5000]

    [memory_strategies.no_strategy]
    type = "no_strategy"
""")

_SAMPLE_MODEL_CONFIG = textwrap.dedent("""\
    [models.gpt-test]
    litellm_name = "openai/gpt-test"
    context_window = 128000
    provider = "test"
""")


def _write_toml_file(tmp_path, name: str, content: str):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# load_configs: model_path=None (default) – no separate registry file
# ---------------------------------------------------------------------------


def test_load_configs_no_model_path_empty_registry(tmp_path):
    """load_configs with model_path=None leaves model_registry empty.

    When no model_path is given and the experiment config contains no [models]
    table, the resulting ExperimentConfig should have an empty model_registry.
    This allows callers to build or inject the registry at a later stage.
    """
    exp_path = _write_toml_file(tmp_path, "config.toml", _MINIMAL_EXPERIMENT_CONFIG)
    cfg = load_configs(exp_path)  # model_path defaults to None
    assert cfg.model_registry == {}


def test_load_configs_models_embedded_in_exp_config(tmp_path):
    """load_configs reads model definitions from [models] inside exp_path.

    When no separate model_path is provided, a [models] table embedded
    directly in the experiment TOML is promoted to model_registry.
    """
    combined = _MINIMAL_EXPERIMENT_CONFIG + _SAMPLE_MODEL_CONFIG
    exp_path = _write_toml_file(tmp_path, "config.toml", combined)
    cfg = load_configs(exp_path)
    assert "gpt-test" in cfg.model_registry
    assert cfg.model_registry["gpt-test"].litellm_name == "openai/gpt-test"


def test_load_configs_explicit_model_path(tmp_path):
    """load_configs merges a separate model registry file when model_path is given.

    This is the original two-file usage pattern: the execution repository keeps
    model_config.toml locally and passes it as model_path.
    """
    exp_path = _write_toml_file(tmp_path, "config.toml", _MINIMAL_EXPERIMENT_CONFIG)
    model_path = _write_toml_file(tmp_path, "model_config.toml", _SAMPLE_MODEL_CONFIG)
    cfg = load_configs(exp_path, model_path=model_path)
    assert "gpt-test" in cfg.model_registry


def test_load_configs_missing_exp_path_raises(tmp_path):
    """load_configs raises FileNotFoundError when the experiment config is absent.

    The experiment config is mandatory; a clear error must be raised rather than
    an opaque exception from the TOML parser.
    """
    with pytest.raises(FileNotFoundError, match="experiment config"):
        load_configs(str(tmp_path / "nonexistent.toml"))


def test_load_configs_missing_model_path_raises(tmp_path):
    """load_configs raises FileNotFoundError when an explicit model_path is absent.

    If the caller explicitly provides a model_path that does not exist, a clear
    error is raised rather than silently ignoring it.
    """
    exp_path = _write_toml_file(tmp_path, "config.toml", _MINIMAL_EXPERIMENT_CONFIG)
    with pytest.raises(FileNotFoundError, match="model config"):
        load_configs(exp_path, model_path=str(tmp_path / "missing_model.toml"))
