"""
Server module for uvicorn to import.

This module creates the FastAPI app instance using configuration
from environment variables (set by the CLI).
"""

import os

from src.memorch.api.app import create_app

# Read configuration from environment variables (set by CLI)
exp_config = os.environ.get("MEMORY_API_EXP_CONFIG", "config.toml")
model_config = os.environ.get("MEMORY_API_MODEL_CONFIG", "model_config.toml")
model_key = os.environ.get("MEMORY_API_MODEL_KEY")
memory_key = os.environ.get("MEMORY_API_MEMORY_KEY")

# Create the app instance
app = create_app(
    exp_path=exp_config,
    model_path=model_config,
    model_key=model_key,
    memory_key=memory_key,
)
