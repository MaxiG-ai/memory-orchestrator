"""
Command-line interface for the Memory Orchestrator API.

Provides commands to start the API server with configurable options.
"""

import click
import uvicorn


@click.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind the server to",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind the server to",
)
@click.option(
    "--exp-config",
    default="config.toml",
    help="Path to experiment configuration file",
)
@click.option(
    "--model-config",
    default="model_config.toml",
    help="Path to model registry configuration file",
)
@click.option(
    "--model",
    "model_key",
    default=None,
    help="Override the default model key",
)
@click.option(
    "--memory",
    "memory_key",
    default=None,
    help="Override the default memory strategy",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload for development",
)
def main(
    host: str,
    port: int,
    exp_config: str,
    model_config: str,
    model_key: str | None,
    memory_key: str | None,
    reload: bool,
):
    """
    Start the Memory Orchestrator API server.

    The server exposes Open Responses format endpoints for LLM interactions
    with configurable memory strategies.

    Example usage:
        memory-api --port 8080 --memory truncation
        memory-api --model gpt-4 --memory progressive_summarization
    """
    # Set environment variables for the app factory to pick up
    import os

    os.environ["MEMORY_API_EXP_CONFIG"] = exp_config
    os.environ["MEMORY_API_MODEL_CONFIG"] = model_config
    if model_key:
        os.environ["MEMORY_API_MODEL_KEY"] = model_key
    if memory_key:
        os.environ["MEMORY_API_MEMORY_KEY"] = memory_key

    # Create app factory string for uvicorn
    # Use a factory pattern to allow config via env vars
    uvicorn.run(
        "src.memorch.api.server:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
