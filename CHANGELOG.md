# Changelog

All notable changes to Memory Orchestrator are documented in this file.

## [Unreleased]

### Added

#### Weave Tracing for Dataset Benchmarks
- Integrated Weave tracing throughout the Memory Orchestrator for full visibility into benchmark runs
- Added `@weave.op()` decorators to all internal memory operations:
  - `ACE Strategy`: `Generator.generate()`, `Reflector.reflect()`, `Curator.curate()`
  - `MemoryBank Strategy`: `ingest_tool_outputs()`, `retrieve_and_format()`, `observe_tool_output()`
- Automatic trace nesting: Internal operations become child traces when called within benchmark runners
- Comprehensive documentation in `LLMOrchestrator` and `MemoryProcessor` classes
- New example: `examples/weave_benchmark_runner.py` showing complete BFCL v3-style benchmark with tracing
- Updated README with observability section and trace hierarchy documentation
- 18 integration tests validating Weave decorator placement and attribute logging

**Benefits:**
- Full transparency into dataset benchmark execution
- Metrics tracking for each internal operation (playbook changes, retrieval stats, etc.)
- Automatic performance analysis via Weights & Biases integration
- Non-breaking change: only adds decorators, zero impact on existing code paths

**Usage:**
```python
import weave
weave.init("my-benchmark")

@weave.op()
def run_benchmark(data):
    orchestrator = LLMOrchestrator()
    processor = MemoryProcessor(strategy="ace")
    # ... benchmark logic
```

See `examples/weave_benchmark_runner.py` for complete example.
