"""
Unit tests for the Open Responses API router.

Tests the API endpoint behavior with mocked LLMOrchestrator to verify:
- Request parsing and validation
- Translation between formats
- Response construction
- Error handling

These tests use FastAPI's TestClient to test the HTTP layer without
making actual LLM calls.
"""

import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient


from src.api.router import router, set_orchestrator

from src.llm_orchestrator import CompressionMetadata


@pytest.fixture
def mock_orchestrator():
    """
    Create a mock LLMOrchestrator with pre-configured responses.

    The mock simulates:
    - Model configuration access
    - Memory key and model key attributes
    - generate_with_memory_applied() returning a mock ChatCompletion
    """
    orchestrator = Mock()
    orchestrator.active_model_key = "test-model"
    orchestrator.active_memory_key = "truncation"

    # Mock model config
    model_def = Mock()
    model_def.litellm_name = "gpt-4"
    orchestrator.get_model_config.return_value = model_def

    # Mock memory processor
    orchestrator.memory_processor = Mock()

    return orchestrator


@pytest.fixture
def mock_compression_metadata():
    """
    Create a mock CompressionMetadata with default values.

    Simulates metadata returned by generate_with_memory_applied()
    when return_metadata=True.
    """
    return CompressionMetadata(
        input_token_count=100,
        compressed_token_count=80,
        compression_ratio=0.8,
        memory_method="truncation",
        processing_time_ms=15.5,
        loop_detected=False,
        strategy_metadata={},
    )


@pytest.fixture
def mock_response():
    """
    Create a mock ChatCompletion response structure.

    Simulates the response from litellm.completion() with:
    - A single choice containing an assistant message
    - Usage statistics
    """
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.role = "assistant"
    response.choices[0].message.content = "Hello! How can I help you?"
    response.choices[0].message.tool_calls = None

    response.usage = Mock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 8
    response.usage.total_tokens = 18

    return response


@pytest.fixture
def client(mock_orchestrator, mock_response, mock_compression_metadata):
    """
    Create a TestClient with mocked orchestrator.

    Configures the orchestrator to return a tuple of (response, metadata)
    to match the return_metadata=True behavior.
    """
    # Return tuple of (response, metadata) to match return_metadata=True behavior
    mock_orchestrator.generate_with_memory_applied.return_value = (
        mock_response,
        mock_compression_metadata,
    )

    set_orchestrator(mock_orchestrator)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    yield TestClient(app)


class TestCreateResponse:
    """Tests for the POST /v1/responses endpoint."""

    def test_simple_string_input_returns_response(self, client, mock_orchestrator):
        """
        Verify that a simple string input is converted to a user message
        and produces a valid Open Responses response.

        The string input should be wrapped as a single user message before
        being passed to the orchestrator.
        """
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-4",
                "input": "Hello, world!",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert data["model"] == "gpt-4"
        assert len(data["output"]) > 0

        # Verify orchestrator was called with user message
        call_args = mock_orchestrator.generate_with_memory_applied.call_args
        messages = call_args.kwargs.get("input_messages") or call_args.args[0]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, world!"

    def test_items_input_converts_correctly(self, client, mock_orchestrator):
        """
        Verify that a list of items is converted to chat-completions format.

        Tests the full translation path: Open Responses items -> chat messages
        -> orchestrator -> chat response -> Open Responses output.
        """
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-4",
                "input": [
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": "You are helpful."}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hi!"}],
                    },
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

        # Verify both messages were passed
        call_args = mock_orchestrator.generate_with_memory_applied.call_args
        messages = call_args.kwargs.get("input_messages") or call_args.args[0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_response_includes_debug_info(self, client):
        """
        Verify that debug_info is populated with memory processing details.

        Debug info should include the memory method, token counts,
        compression ratio, and processing time.
        """
        response = client.post(
            "/v1/responses",
            json={"model": "gpt-4", "input": "Test"},
        )

        assert response.status_code == 200
        data = response.json()

        debug = data.get("debug_info")
        assert debug is not None
        assert debug["memory_method"] == "truncation"
        assert "input_token_count" in debug
        assert "compressed_token_count" in debug
        assert "compression_ratio" in debug
        assert "processing_time_ms" in debug

    def test_response_includes_usage_statistics(self, client):
        """
        Verify that token usage statistics are included in the response.

        Usage should reflect the prompt and completion tokens from
        the LLM response.
        """
        response = client.post(
            "/v1/responses",
            json={"model": "gpt-4", "input": "Test"},
        )

        assert response.status_code == 200
        data = response.json()

        usage = data.get("usage")
        assert usage is not None
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 8
        assert usage["total_tokens"] == 18

    def test_tools_passed_to_orchestrator(self, client, mock_orchestrator):
        """
        Verify that tools are converted and passed to the orchestrator.

        Open Responses FunctionTool format should be converted to
        chat-completions tool format.
        """
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-4",
                "input": "What's the weather?",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    }
                ],
            },
        )

        assert response.status_code == 200

        # Verify tools were passed
        call_args = mock_orchestrator.generate_with_memory_applied.call_args
        tools = call_args.kwargs.get("tools")
        assert tools is not None
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "get_weather"

    def test_temperature_and_max_tokens_passed(self, client, mock_orchestrator):
        """
        Verify that generation parameters are forwarded to the orchestrator.

        Temperature and max_output_tokens should be passed as kwargs
        to generate_with_memory_applied.
        """
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-4",
                "input": "Test",
                "temperature": 0.7,
                "max_output_tokens": 100,
            },
        )

        assert response.status_code == 200

        call_args = mock_orchestrator.generate_with_memory_applied.call_args
        assert call_args.kwargs.get("temperature") == 0.7
        assert call_args.kwargs.get("max_tokens") == 100


class TestToolCallResponse:
    """Tests for responses containing tool calls."""

    def test_tool_calls_in_response_converted_to_items(self, mock_orchestrator):
        """
        Verify that assistant messages with tool_calls are converted
        to separate FunctionCallItem entries in the output.

        An assistant message with tool_calls should produce:
        1. A MessageItem for the assistant
        2. FunctionCallItem entries for each tool call
        """
        # Create mock response with tool calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = None

        # Create proper tool call mock with actual string values
        tool_call = Mock()
        tool_call.id = "call_123"
        tool_call.function = Mock()
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"location": "NYC"}'
        mock_response.choices[0].message.tool_calls = [tool_call]

        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 8
        mock_response.usage.total_tokens = 18

        # Create mock compression metadata
        mock_metadata = CompressionMetadata(
            input_token_count=100,
            compressed_token_count=80,
            compression_ratio=0.8,
            memory_method="truncation",
            processing_time_ms=15.5,
        )

        # Return tuple of (response, metadata) to match return_metadata=True behavior
        mock_orchestrator.generate_with_memory_applied.return_value = (
            mock_response,
            mock_metadata,
        )

        set_orchestrator(mock_orchestrator)
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/v1/responses",
            json={"model": "gpt-4", "input": "What's the weather in NYC?"},
        )

        assert response.status_code == 200
        data = response.json()

        # Should have assistant message + function call
        output = data["output"]
        assert len(output) >= 1

        # Find function_call item
        function_calls = [o for o in output if o["type"] == "function_call"]
        assert len(function_calls) == 1
        assert function_calls[0]["call_id"] == "call_123"
        assert function_calls[0]["name"] == "get_weather"


class TestErrorHandling:
    """Tests for error handling in the API."""

    def test_orchestrator_error_returns_failed_status(self, client, mock_orchestrator):
        """
        Verify that orchestrator exceptions result in a failed response
        with error details, not an HTTP error.

        The API should catch exceptions and return a proper Open Responses
        response with status=failed and error information.
        """
        mock_orchestrator.generate_with_memory_applied.side_effect = Exception(
            "Model API error"
        )

        response = client.post(
            "/v1/responses",
            json={"model": "gpt-4", "input": "Test"},
        )

        assert response.status_code == 200  # Still returns 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] is not None
        assert "Model API error" in data["error"]["message"]

    def test_invalid_request_returns_422(self, client):
        """
        Verify that malformed requests return HTTP 422.

        Missing required fields should be caught by Pydantic validation
        before reaching the endpoint handler.
        """
        response = client.post(
            "/v1/responses",
            json={"input": "Test"},  # Missing required 'model'
        )

        assert response.status_code == 422


class TestToolChoiceConversion:
    """Tests for tool_choice parameter handling."""

    def test_tool_choice_none_passed_correctly(self, client, mock_orchestrator):
        """
        Verify tool_choice='none' is passed to orchestrator.
        """
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-4",
                "input": "Test",
                "tool_choice": "none",
            },
        )

        assert response.status_code == 200
        call_args = mock_orchestrator.generate_with_memory_applied.call_args
        assert call_args.kwargs.get("tool_choice") == "none"

    def test_tool_choice_required_passed_correctly(self, client, mock_orchestrator):
        """
        Verify tool_choice='required' is passed to orchestrator.
        """
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-4",
                "input": "Test",
                "tool_choice": "required",
                "tools": [{"type": "function", "name": "test_fn"}],
            },
        )

        assert response.status_code == 200
        call_args = mock_orchestrator.generate_with_memory_applied.call_args
        assert call_args.kwargs.get("tool_choice") == "required"
