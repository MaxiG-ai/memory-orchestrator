# memory-orchestrator

LLM wrapper that applies memory operations. Used in my master thesis on investigating memory methods for their impact on tool use.

## Features

- **Multiple Memory Strategies**: ACE (Adaptive Context Enhancement) and MemoryBank (Vector-based Retrieval)
- **Flexible LLM Integration**: Works with any LLM via LiteLLM
- **Full Observability**: Integrated Weave tracing for benchmark runs
- **Comprehensive Testing**: 18+ integration tests for tracing validation

## Observability & Tracing

The Memory Orchestrator integrates with [Weave](https://weave.weights-biases.com) for full visibility into benchmark runs (e.g., BFCL v3).

### Quick Start

```python
import weave
from memory_orchestrator import LLMOrchestrator, MemoryProcessor

# Initialize Weave
weave.init("my-benchmark-project")

# Decorate your benchmark runner
@weave.op()
def run_benchmark(benchmark_data):
    orchestrator = LLMOrchestrator()
    processor = MemoryProcessor(strategy="ace")
    
    # Your benchmark logic here
    response = orchestrator.generate_with_memory_applied(
        user_input="...",
        conversation_history=[...],
        memory_processor=processor,
    )
    return response
```

### Trace Hierarchy

When you run a benchmark with Weave tracing, you get a complete hierarchy:

```
run_benchmark
├── run_task (task_id, initial_state)
│   ├── run_step (turn_number, user_message)
│   │   ├── Generator.generate() [logs reasoning_trace, bullet_ids_used]
│   │   ├── Reflector.reflect() [logs reflection_text, bullet_tags]
│   │   └── Curator.curate() [logs operations, playbook_diffs]
│   │
│   └── run_step (turn_number, user_message)
│       └── ...
│
└── run_task (task_id, initial_state)
    └── ...
```

**Automatically Traced Operations:**
- `Generator.generate()` - Playbook-based reasoning (ACE strategy)
- `Reflector.reflect()` - Reflection on generated content
- `Curator.curate()` - Playbook updates and operations
- `ingest_tool_outputs()` - Tool output ingestion (MemoryBank strategy)
- `retrieve_and_format()` - Memory retrieval and formatting
- `observe_tool_output()` - Tool output observation

### Example: BFCL v3 Benchmark

See `examples/weave_benchmark_runner.py` for a complete example of running the BFCL v3 benchmark with Weave tracing:

```bash
python examples/weave_benchmark_runner.py
```

This example shows:
- How to initialize Weave with your project name
- How to structure your benchmark runner with proper trace hierarchy
- How to log task-level and step-level metrics
- How internal operations automatically become child traces

After running, view your traces at: `https://weave.weights-biases.com/projects/my-benchmark-project`

### Logged Attributes by Operation

**ACE Strategy (Playbook-based):**
- `Generator.generate()`: `playbook_tokens`, `context_length`, `has_reflection`, `reasoning_trace`, `bullet_ids_used`
- `Reflector.reflect()`: `bullets_count`, `reflection_text`, `bullet_tags`
- `Curator.curate()`: `playbook_tokens_before/after`, `bullet_count_before/after`, `operations` (add/update/delete)

**MemoryBank Strategy (Vector-based):**
- `ingest_tool_outputs()`: `num_tool_outputs`, `step_id`, `trace_ids`, `total_chars_ingested`
- `retrieve_and_format()`: `query`, `top_k`, `num_retrieved`, `retrieved_records`
- `observe_tool_output()`: `tool_name`, `output_chars`, `summary`

### Integration with Weights & Biases

Weave traces are automatically sent to Weights & Biases. To view them:

1. Create a W&B account at https://wandb.ai
2. Initialize Weave with your entity name: `weave.init("project", entity="your-entity")`
3. Visit `https://weave.weights-biases.com/projects/your-project`

Traces include:
- Complete trace hierarchy with automatic nesting
- Metrics and operation tracking (not full conversation history)
- Timing information for performance analysis
- Input/output for debugging
