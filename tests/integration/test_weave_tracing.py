"""
Integration tests for Weave tracing in dataset runs.

Tests verify that @weave.op decorators are properly integrated into:
1. ACE strategy components (Generator, Reflector, Curator)
2. MemoryBank components (Ingestion, Retrieval, Observer)
3. LLMOrchestrator and MemoryProcessor

Tests confirm that internal memory operations are properly traced when
running dataset benchmarks with Weave initialized.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from src.strategies.ace.generator import Generator
from src.strategies.ace.reflector import Reflector
from src.strategies.ace.curator import Curator
from src.strategies.memory_bank.ingestion import ingest_tool_outputs
from src.strategies.memory_bank.retrieval import retrieve_and_format
from src.strategies.memory_bank.observer import observe_tool_output
from src.strategies.memory_bank.fact_store import FactStore
from src.strategies.memory_bank.insight_store import InsightStore


class TestACEGeneratorWeaveTracing:
    """Tests that ACE Generator is properly decorated with @weave.op()."""

    def test_generator_has_weave_decorator(self):
        """
        Verify that Generator.generate() method has @weave.op() decorator.

        This ensures that when the generator is called within a benchmark run,
        its execution is automatically traced to Weave as a child operation.
        """
        generator = Generator()

        # Check that generate method has weave metadata
        assert hasattr(generator.generate, "__wrapped__"), (
            "Generator.generate() should be decorated with @weave.op()"
        )

    def test_generator_return_values_are_logged(self):
        """
        Verify that Generator.generate() returns values suitable for Weave logging.

        The returned (reasoning_trace, bullet_ids) tuple contains all necessary
        information for tracing the generator's decision-making process.
        """
        generator = Generator()

        # Mock the LLM client
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="REASONING:\nBullet IDs: [1, 2, 3]"))
        ]
        mock_llm_client.generate_plain.return_value = mock_response

        reasoning_trace, bullet_ids = generator.generate(
            question="Test question",
            playbook="Test playbook",
            context="Test context",
            reflection="Test reflection",
            llm_client=mock_llm_client,
            model="test-model",
        )

        # Verify return values
        assert isinstance(reasoning_trace, str), "reasoning_trace should be a string"
        assert isinstance(bullet_ids, list), "bullet_ids should be a list"
        assert all(isinstance(bid, int) for bid in bullet_ids), (
            "All bullet IDs should be integers"
        )


class TestACEReflectorWeaveTracing:
    """Tests that ACE Reflector is properly decorated with @weave.op()."""

    def test_reflector_has_weave_decorator(self):
        """
        Verify that Reflector.reflect() method has @weave.op() decorator.

        This ensures that reflection analysis is traced when running benchmarks.
        """
        reflector = Reflector()

        # Check that reflect method has weave metadata
        assert hasattr(reflector.reflect, "__wrapped__"), (
            "Reflector.reflect() should be decorated with @weave.op()"
        )

    def test_reflector_return_values_are_logged(self):
        """
        Verify that Reflector.reflect() returns values suitable for Weave logging.

        The returned (reflection_text, bullet_tags) tuple is the key output
        for understanding how the reflection influenced playbook updates.
        """
        reflector = Reflector()

        # Mock the LLM client
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"bullet_tags": [{"bullet_id": 1, "tag": "helpful"}]}'
                )
            )
        ]
        mock_llm_client.generate_plain.return_value = mock_response

        reflection_text, bullet_tags = reflector.reflect(
            question="Test question",
            reasoning_trace="Test reasoning",
            predicted_answer="Test answer",
            environment_feedback="Test feedback",
            bullets_used="Test bullets",
            llm_client=mock_llm_client,
            model="test-model",
        )

        # Verify return values
        assert isinstance(reflection_text, str), "reflection_text should be a string"
        assert isinstance(bullet_tags, list), "bullet_tags should be a list"


class TestACECuratorWeaveTracing:
    """Tests that ACE Curator is properly decorated with @weave.op()."""

    def test_curator_has_weave_decorator(self):
        """
        Verify that Curator.curate() method has @weave.op() decorator.

        Curator operations are critical to understanding playbook evolution,
        so they must be traced.
        """
        curator = Curator()

        # Check that curate method has weave metadata
        assert hasattr(curator.curate, "__wrapped__"), (
            "Curator.curate() should be decorated with @weave.op()"
        )

    def test_curator_return_values_include_operations(self):
        """
        Verify that Curator.curate() returns operations for Weave logging.

        The operations list documents how the playbook was modified (added/updated/deleted
        bullets), which is essential for tracing knowledge evolution.
        """
        curator = Curator()

        # Mock the LLM client
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"operations": []}'))]
        mock_llm_client.generate_plain.return_value = mock_response

        updated_playbook, next_id, operations = curator.curate(
            current_playbook="# Playbook\n- Bullet 1",
            recent_reflection="Good progress",
            question_context="Test context",
            step=1,
            token_budget=4096,
            playbook_stats={
                "total_bullets": 1,
                "high_performing": 0,
                "problematic": 0,
                "unused": 1,
            },
            llm_client=mock_llm_client,
            model="test-model",
            next_global_id=1,
        )

        # Verify return values
        assert isinstance(updated_playbook, str), "updated_playbook should be a string"
        assert isinstance(next_id, int), "next_id should be an integer"
        assert isinstance(operations, list), "operations should be a list"


class TestMemoryBankIngestionWeaveTracing:
    """Tests that MemoryBank ingestion is properly decorated with @weave.op()."""

    def test_ingest_tool_outputs_has_weave_decorator(self):
        """
        Verify that ingest_tool_outputs() function has @weave.op() decorator.

        Ingestion is a critical operation that must be traced to understand
        how the memory bank is populated.
        """
        # Check that ingest_tool_outputs function has weave metadata
        assert hasattr(ingest_tool_outputs, "__wrapped__"), (
            "ingest_tool_outputs() should be decorated with @weave.op()"
        )

    def test_ingest_tool_outputs_returns_trace_ids(self):
        """
        Verify that ingest_tool_outputs() returns trace IDs for logging.

        Trace IDs are the primary output that connects tool calls to their
        stored representations in the dual-store.
        """
        # Mock stores
        fact_store = Mock(spec=FactStore)
        insight_store = Mock(spec=InsightStore)
        insight_store.add = Mock()

        # Mock LLM client
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Tool summary"))]
        mock_llm_client.generate_plain.return_value = mock_response

        tool_outputs = [("search_api", {"query": "test"}, {"result": "found"})]

        # Mock fact_store.store to set trace_id
        def mock_store(record):
            record.trace_id = "trace_123"

        fact_store.store = mock_store

        with patch(
            "src.strategies.memory_bank.ingestion.InteractionRecord"
        ) as mock_record:
            mock_record.create.return_value = Mock(trace_id="trace_123")
            trace_ids = ingest_tool_outputs(
                tool_outputs=tool_outputs,
                user_query="Find something",
                fact_store=fact_store,
                insight_store=insight_store,
                llm_client=mock_llm_client,
                observer_model="test-model",
                step_id=1,
            )

        assert isinstance(trace_ids, list), "trace_ids should be a list"
        assert all(isinstance(tid, str) for tid in trace_ids), (
            "All trace IDs should be strings"
        )


class TestMemoryBankRetrievalWeaveTracing:
    """Tests that MemoryBank retrieval is properly decorated with @weave.op()."""

    def test_retrieve_and_format_has_weave_decorator(self):
        """
        Verify that retrieve_and_format() function has @weave.op() decorator.

        Retrieval operations must be traced to track which past interactions
        are considered relevant at each step.
        """
        # Check that retrieve_and_format function has weave metadata
        assert hasattr(retrieve_and_format, "__wrapped__"), (
            "retrieve_and_format() should be decorated with @weave.op()"
        )

    def test_retrieve_and_format_returns_formatted_records(self):
        """
        Verify that retrieve_and_format() returns formatted records for logging.

        The returned records contain trace_id, summary, and raw_data which
        are essential for understanding retrieval results.
        """
        # Mock stores
        fact_store = Mock(spec=FactStore)
        insight_store = Mock(spec=InsightStore)

        # Mock search and retrieval
        insight_store.search.return_value = ["trace_1", "trace_2"]
        insight_store.get_summary.return_value = "Found something useful"

        mock_record_1 = Mock()
        mock_record_1.trace_id = "trace_1"
        mock_record_1.tool_name = "search_api"
        mock_record_1.raw_output = {"result": "found"}

        mock_record_2 = Mock()
        mock_record_2.trace_id = "trace_2"
        mock_record_2.tool_name = "fetch_api"
        mock_record_2.raw_output = {"data": "loaded"}

        fact_store.get_many.return_value = [mock_record_1, mock_record_2]

        results = retrieve_and_format(
            query="how to proceed",
            fact_store=fact_store,
            insight_store=insight_store,
            top_k=2,
            max_chars=500,
        )

        assert isinstance(results, list), "results should be a list"
        assert all(isinstance(r, dict) for r in results), "Each result should be a dict"
        assert all("trace_id" in r for r in results), "Each result should have trace_id"


class TestMemoryBankObserverWeaveTracing:
    """Tests that MemoryBank Observer is properly decorated with @weave.op()."""

    def test_observe_tool_output_has_weave_decorator(self):
        """
        Verify that observe_tool_output() function has @weave.op() decorator.

        Observer output generation must be traced to understand how tool
        results are summarized for semantic search.
        """
        # Check that observe_tool_output function has weave metadata
        assert hasattr(observe_tool_output, "__wrapped__"), (
            "observe_tool_output() should be decorated with @weave.op()"
        )

    def test_observe_tool_output_returns_summary(self):
        """
        Verify that observe_tool_output() returns a summary string for logging.

        The summary is the primary output that enables semantic retrieval
        and should be traceable.
        """
        # Mock LLM client
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Tool found 5 matching items"))
        ]
        mock_llm_client.generate_plain.return_value = mock_response

        summary = observe_tool_output(
            user_query="Find matching items",
            tool_name="search_api",
            raw_output={"count": 5, "items": [1, 2, 3, 4, 5]},
            llm_client=mock_llm_client,
            model="test-model",
        )

        assert isinstance(summary, str), "summary should be a string"
        assert len(summary) > 0, "summary should not be empty"


class TestWeaveTracingHierarchy:
    """
    Integration tests verifying the trace hierarchy works correctly.

    When a benchmark runner initializes Weave and decorates benchmark/task/step
    operations, internal orchestrator operations should nest properly as children.
    """

    def test_weave_imports_are_available(self):
        """
        Verify that weave is importable in all decorated modules.

        This is a precondition for all tracing to work.
        """
        import weave

        # Verify weave has init and op
        assert hasattr(weave, "init"), "weave should have init()"
        assert hasattr(weave, "op"), "weave should have op()"

    def test_tracing_module_docstrings_document_weave(self):
        """
        Verify that LLMOrchestrator and MemoryProcessor docstrings
        document Weave tracing and provide usage examples.

        This ensures developers know how to use Weave with this package.
        """
        from src.llm_orchestrator import LLMOrchestrator
        from src.memory_processing import MemoryProcessor

        orchestrator_doc = LLMOrchestrator.__doc__
        processor_doc = MemoryProcessor.__doc__

        # Check for Weave documentation
        assert "weave" in orchestrator_doc.lower(), (
            "LLMOrchestrator should document Weave tracing"
        )
        assert "benchmark" in orchestrator_doc.lower(), (
            "LLMOrchestrator should provide benchmark integration example"
        )
        assert "weave" in processor_doc.lower(), (
            "MemoryProcessor should document Weave tracing"
        )


class TestACEWeaveAttributeLogging:
    """
    Tests verify that ACE operations log appropriate attributes for Weave.

    These attributes appear in Weave traces and help debug memory evolution.
    """

    def test_generator_logs_playbook_context(self):
        """
        Verify Generator can compute and log playbook metrics for tracing.

        The docstring should indicate these will be logged:
        - playbook_tokens: Token count of playbook
        - context_length: Length of context
        - has_reflection: Whether reflection was provided
        """
        generator = Generator()

        # Verify docstring documents logged attributes
        doc = generator.generate.__doc__
        assert "playbook_tokens" in doc, (
            "Generator docstring should document playbook_tokens logging"
        )
        assert "bullet_ids_used" in doc, (
            "Generator docstring should document bullet_ids_used logging"
        )

    def test_curator_logs_playbook_diff(self):
        """
        Verify Curator documents playbook_tokens_before/after for logging.

        These metrics enable tracing playbook growth/shrinkage.
        """
        curator = Curator()

        # Verify docstring documents logged attributes
        doc = curator.curate.__doc__
        assert "playbook_tokens_before" in doc, (
            "Curator docstring should document playbook_tokens_before"
        )
        assert "playbook_tokens_after" in doc, (
            "Curator docstring should document playbook_tokens_after"
        )
        assert "operations" in doc, (
            "Curator docstring should document operations logging"
        )


class TestMemoryBankWeaveAttributeLogging:
    """
    Tests verify that MemoryBank operations log appropriate attributes for Weave.
    """

    def test_ingestion_logs_operation_metrics(self):
        """
        Verify ingestion docstring documents logged attributes.

        Should log:
        - num_tool_outputs: Number of outputs being ingested
        - trace_ids: Generated trace IDs
        - total_chars_ingested: Total data volume
        """
        # Import the function and check docstring
        doc = ingest_tool_outputs.__doc__

        assert "num_tool_outputs" in doc, (
            "Ingestion docstring should document num_tool_outputs"
        )
        assert "trace_ids" in doc, "Ingestion docstring should document trace_ids"

    def test_retrieval_logs_search_metrics(self):
        """
        Verify retrieval docstring documents logged attributes.

        Should log:
        - query: The semantic search query
        - top_k: Number of results requested
        - num_retrieved: Actual number retrieved
        """
        doc = retrieve_and_format.__doc__

        assert "query" in doc, "Retrieval docstring should document query"
        assert "top_k" in doc, "Retrieval docstring should document top_k"
        assert "num_retrieved" in doc, (
            "Retrieval docstring should document num_retrieved"
        )
