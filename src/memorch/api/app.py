"""
FastAPI application factory for the Open Responses API.

Creates a configured FastAPI application with the responses endpoint
and optional health/status endpoints.
"""

from fastapi import FastAPI

from memorch.api.router import router, set_orchestrator
from memorch.llm_orchestrator import LLMOrchestrator


def create_app(
    exp_path: str = "config.toml",
    model_path: str = "model_config.toml",
    model_key: str | None = None,
    memory_key: str | None = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        exp_path: Path to experiment configuration file
        model_path: Path to model registry configuration file
        model_key: Override default model selection
        memory_key: Override default memory strategy

    Returns:
        Configured FastAPI application ready to serve requests
    """
    app = FastAPI(
        title="Memory Orchestrator API",
        description="Open Responses API for LLM memory evaluation",
        version="0.1.0",
    )

    # Initialize orchestrator
    orchestrator = LLMOrchestrator(exp_path=exp_path, model_path=model_path)

    # Apply overrides if provided
    if model_key or memory_key:
        orchestrator.set_active_context(
            model_key=model_key or orchestrator.active_model_key,
            memory_key=memory_key or orchestrator.active_memory_key,
        )

    # Register orchestrator with router
    set_orchestrator(orchestrator)

    # Include the responses router
    app.include_router(router)

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "model": orchestrator.active_model_key,
            "memory_strategy": orchestrator.active_memory_key,
        }

    return app
