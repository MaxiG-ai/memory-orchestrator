"""
FastAPI router for Open Responses API endpoints.

Provides the /v1/responses endpoint that accepts Open Responses format requests,
translates them to chat-completions format for the LLMOrchestrator, and returns
responses in Open Responses format.
"""

import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from memorch.api.models import (
    OpenResponsesRequest,
    OpenResponsesResponse,
    ResponseStatus,
    Usage,
    DebugInfo,
    FunctionTool,
    Item,
)
from memorch.api.translator import chat_completions_to_items, items_to_chat_completions
from memorch.llm_orchestrator import LLMOrchestrator
from memorch.utils.logger import get_logger

logger = get_logger("API")

router = APIRouter(prefix="/v1", tags=["responses"])

# Global orchestrator instance (set by create_app)
_orchestrator: Optional[LLMOrchestrator] = None


def set_orchestrator(orchestrator: LLMOrchestrator) -> None:
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator


def get_orchestrator() -> LLMOrchestrator:
    """Dependency to get the orchestrator instance."""
    if _orchestrator is None:
        raise HTTPException(
            status_code=500,
            detail="Orchestrator not initialized. Call set_orchestrator() first.",
        )
    return _orchestrator


def _convert_tools_to_chat_format(tools: list[FunctionTool]) -> list[dict]:
    """Convert Open Responses tools to chat-completions format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters or {},
                "strict": tool.strict,
            },
        }
        for tool in tools
    ]


def _convert_tool_choice(tool_choice):
    """Convert Open Responses tool_choice to chat-completions format.

    Returns str for simple choices (auto, none, required) or dict for specific function.
    """
    if tool_choice is None:
        return "auto"
    # Handle enum values
    if hasattr(tool_choice, "value"):
        return tool_choice.value
    # Handle SpecificFunctionChoice
    if hasattr(tool_choice, "name"):
        return {"type": "function", "function": {"name": tool_choice.name}}
    return str(tool_choice)


@router.post("/responses", response_model=OpenResponsesResponse)
async def create_response(
    request: OpenResponsesRequest,
    orchestrator: LLMOrchestrator = Depends(get_orchestrator),
) -> OpenResponsesResponse:
    """
    Create a model response using the Open Responses format.

    Accepts input in Open Responses format (items), applies the configured
    memory strategy, and returns the response in Open Responses format.
    """
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())

    try:
        # Convert input to chat-completions format
        if isinstance(request.input, str):
            # Simple string input -> single user message
            messages = [{"role": "user", "content": request.input}]
        else:
            # List of items -> convert to messages
            messages = items_to_chat_completions(request.input)

        # Prepare tools if provided
        tools = None
        if request.tools:
            tools = _convert_tools_to_chat_format(request.tools)

        # Prepare tool_choice
        tool_choice = _convert_tool_choice(request.tool_choice)

        # Build kwargs from request parameters
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens

        # Call the orchestrator with metadata to get compression info
        response, compression_metadata = orchestrator.generate_with_memory_applied(
            input_messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            return_metadata=True,
            **kwargs,
        )

        # Extract the response content
        output_items: list[Item] = []
        if not response.choices:
            raise ValueError("LLM response contained no choices")
        choice = response.choices[0]
        message = choice.message

        # Convert response to Open Responses items
        response_messages = [
            {
                "role": message.role,
                "content": message.content,
                "tool_calls": (
                    [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ]
                    if message.tool_calls
                    else None
                ),
            }
        ]
        output_items = chat_completions_to_items(response_messages)

        # Build usage info
        usage = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        # Build debug info from compression metadata
        debug_info = DebugInfo(
            memory_method=compression_metadata.memory_method,
            input_token_count=compression_metadata.input_token_count,
            compressed_token_count=compression_metadata.compressed_token_count,
            compression_ratio=compression_metadata.compression_ratio,
            strategy_metadata=compression_metadata.strategy_metadata,
            processing_time_ms=compression_metadata.processing_time_ms,
            loop_detection=compression_metadata.loop_detected,
        )

        return OpenResponsesResponse(
            id=response_id,
            created_at=created_at,
            completed_at=int(time.time()),
            status=ResponseStatus.completed,
            model=request.model,
            output=output_items,
            usage=usage,
            tools=request.tools,
            tool_choice=request.tool_choice,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            parallel_tool_calls=request.parallel_tool_calls,
            metadata=request.metadata,
            debug_info=debug_info,
        )

    except Exception as e:
        logger.error(f"Request failed: {e}")
        return OpenResponsesResponse(
            id=response_id,
            created_at=created_at,
            completed_at=int(time.time()),
            status=ResponseStatus.failed,
            model=request.model,
            output=[],
            error={"type": "api_error", "message": str(e)},
        )
