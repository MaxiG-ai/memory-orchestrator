"""
Weave Tracing Integration Example for Dataset Benchmarks

This example demonstrates how to integrate Weave tracing with the Memory
Orchestrator for observability in dataset benchmark runs (e.g., BFCL v3).

Key features:
- Full trace hierarchy: benchmark -> task -> step -> internal operations
- Automatic nesting of internal operations (Generator, Reflector, Curator, etc.)
- Metrics-only logging (not full conversation history)
- Integration with Weights & Biases for visualization

Example trace hierarchy:
    run_benchmark
        ├── run_task (task_id, initial_state)
        │   ├── run_step (turn_number, user_message)
        │   │   ├── Generator.generate() [logs reasoning_trace, bullet_ids]
        │   │   ├── Reflector.reflect() [logs reflection_text, bullet_tags]
        │   │   ├── Curator.curate() [logs operations, playbook diffs]
        │   │   └── (OR) ingest_tool_outputs, retrieve_and_format, observe_tool_output
        │   │
        │   └── run_step (turn_number, user_message)
        │       └── ...
        │
        └── run_task (task_id, initial_state)
            └── ...
"""

import weave
from typing import Any, Optional
from memory_orchestrator import LLMOrchestrator, MemoryProcessor
from memory_orchestrator.schemas import ConversationMessage


class WeaveBenchmarkRunner:
    """
    Wrapper for running benchmarks with full Weave tracing.

    Usage:
        runner = WeaveBenchmarkRunner(project_name="my-benchmark")
        runner.run_benchmark(benchmark_data)
    """

    def __init__(self, project_name: str, entity: Optional[str] = None):
        """
        Initialize Weave benchmark runner.

        Args:
            project_name: Name of the Weave project (appears in W&B dashboard)
            entity: Optional W&B entity/team name
        """
        self.project_name = project_name
        weave.init(project_name, entity=entity)

    @weave.op()
    def run_benchmark(self, benchmark_data: list[dict]) -> dict:
        """
        Run complete benchmark with tracing.

        Args:
            benchmark_data: List of benchmark tasks

        Returns:
            Benchmark results with trace information
        """
        results = {
            "tasks": [],
            "metadata": {
                "project": self.project_name,
                "total_tasks": len(benchmark_data),
            },
        }

        for task_idx, task in enumerate(benchmark_data):
            task_result = self.run_task(task, task_idx)
            results["tasks"].append(task_result)

        return results

    @weave.op()
    def run_task(self, task: dict, task_id: int) -> dict:
        """
        Run single task (e.g., one tool use scenario).

        Logged attributes:
        - task_id: Task identifier
        - initial_state: Starting conversation state

        Args:
            task: Task configuration
            task_id: Task identifier for logging

        Returns:
            Task result with trace reference
        """
        # Initialize conversation for this task
        conversation: list[ConversationMessage] = []
        memory_processor = MemoryProcessor(strategy="ace")  # or "memory_bank"

        task_result = {
            "task_id": task_id,
            "steps": [],
            "final_answer": None,
        }

        # Process each turn in the task
        for turn_idx, turn in enumerate(task.get("turns", [])):
            step_result = self.run_step(
                turn_idx,
                turn,
                conversation,
                memory_processor,
            )
            task_result["steps"].append(step_result)

        task_result["final_answer"] = conversation[-1].content if conversation else None
        return task_result

    @weave.op()
    def run_step(
        self,
        turn_number: int,
        turn_data: dict,
        conversation: list[ConversationMessage],
        memory_processor: MemoryProcessor,
    ) -> dict:
        """
        Run single step (one turn in a conversation).

        This is where internal Memory Orchestrator operations become child traces:
        - Generator.generate() traces captured as child operations
        - Reflector.reflect() traces captured as child operations
        - Curator.curate() traces captured as child operations
        - retrieve_and_format() or ingest_tool_outputs() traces captured

        Logged attributes:
        - turn_number: Turn index in conversation
        - user_message: User input message
        - conversation_length: Current conversation length

        Args:
            turn_number: Turn index
            turn_data: Turn input/expected output
            conversation: Conversation history
            memory_processor: Memory processor instance

        Returns:
            Step result with generated answer
        """
        user_message = turn_data.get("user_message", "")

        # Add user message to conversation
        conversation.append(ConversationMessage(role="user", content=user_message))

        # Generate response using orchestrator with memory
        orchestrator = LLMOrchestrator()
        response = orchestrator.generate_with_memory_applied(
            user_input=user_message,
            conversation_history=conversation,
            memory_processor=memory_processor,
        )

        # Add assistant response to conversation
        conversation.append(ConversationMessage(role="assistant", content=response))

        # Prepare step result
        step_result = {
            "turn_number": turn_number,
            "user_message": user_message,
            "assistant_response": response,
            "conversation_length": len(conversation),
            "expected_output": turn_data.get("expected_output"),
        }

        return step_result


# Example usage
if __name__ == "__main__":
    # Sample BFCL-style benchmark data
    benchmark_data = [
        {
            "task_id": "bfcl_v3_task_001",
            "turns": [
                {
                    "user_message": "What's the weather in San Francisco?",
                    "expected_output": "I retrieved the weather for San Francisco...",
                },
                {
                    "user_message": "And in London?",
                    "expected_output": "I retrieved the weather for London...",
                },
            ],
        },
        {
            "task_id": "bfcl_v3_task_002",
            "turns": [
                {
                    "user_message": "Get me the latest news",
                    "expected_output": "Here are the latest news articles...",
                },
            ],
        },
    ]

    # Run benchmark with Weave tracing
    runner = WeaveBenchmarkRunner(
        project_name="bfcl-v3-benchmark",
        entity=None,  # Set to your W&B entity if desired
    )

    results = runner.run_benchmark(benchmark_data)

    print(f"Benchmark completed: {results['metadata']['total_tasks']} tasks")
    print(
        f"View traces at: https://weave.weights-biases.com/projects/{runner.project_name}"
    )
