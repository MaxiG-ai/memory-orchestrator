# Specs for Progressive Summarization

Here are some headline facts about Progressive Summarization:

- gets called at threshold value set in `config.toml`
- Implements a summary of entire conversation, leaving only original user message intact.
- Fallback model is GPT-4.1

## Files used

- `src/memorch/memory_processing.py` calls this technique.
- `src/memorch/strategies/progressive_summarization/prog_sum.py` implements the actual summarization. Depends on `utils/trace_processing.py` and `utils/llm_helpers.py`
- loads the summarization prompt from `src/memorch/strategies/progressive_summarization/prog_sum.prompt.md`
