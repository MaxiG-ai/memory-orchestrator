from memorch.utils.config import ExperimentConfig, MemoryDef


def test_memory_def_accepts_auto_compact_threshold():
    definition = MemoryDef(
        type="progressive_summarization",
        auto_compact_threshold=4000,
        summarizer_model="gpt-5-mini",
    )

    assert definition.auto_compact_threshold == 4000
    assert definition.summarizer_model == "gpt-5-mini"


def _minimal_experiment_config(**overrides) -> dict:
    """Return the minimum valid ExperimentConfig dict, with optional field overrides."""
    base = {
        "experiment_name": "test",
        "results_dir": "results",
        "log_dir": "logs",
        "logging_level": "INFO",
        "input_file": "data.jsonl",
        "enabled_models": ["gpt-4-1-mini"],
        "enabled_memory_methods": ["no_strategy"],
        "compact_thresholds": [5000],
        "memory_strategies": {"no_strategy": {"type": "no_strategy"}},
    }
    base.update(overrides)
    return base


def test_experiment_config_evaluation_model_defaults_to_gpt4_mini():
    """
    ExperimentConfig must default evaluation_model to 'gpt-4-1-mini' when the
    field is absent from the TOML.  This preserves backwards compatibility for
    all existing experiment configs that pre-date the field and ensures that
    evaluation/judge LLM calls are never accidentally routed through an
    experimental model just because no explicit override was provided.
    """
    config = ExperimentConfig(**_minimal_experiment_config())
    assert config.evaluation_model == "gpt-4-1-mini"


def test_experiment_config_evaluation_model_can_be_overridden():
    """
    evaluation_model must be overridable via config so operators can pin
    evaluations to a different judge model (e.g. a newer or cheaper model)
    without touching source code.
    """
    config = ExperimentConfig(**_minimal_experiment_config(evaluation_model="gpt-5"))
    assert config.evaluation_model == "gpt-5"


def test_experiment_config_evaluation_model_accepts_none():
    """
    Setting evaluation_model = None must be accepted and signals that
    generate_plain() should fall back to the active benchmarked model.
    This is the explicit opt-out path for callers who intentionally want
    the same model for both benchmarking and evaluation.
    """
    config = ExperimentConfig(**_minimal_experiment_config(evaluation_model=None))
    assert config.evaluation_model is None
