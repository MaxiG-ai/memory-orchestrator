# Utils module for thesis-function-calling

import threading

from .logger import get_logger, set_global_log_level
from .token_count import get_token_count
from .trace_history import TraceHistoryEntry, _serialize_message

# Global lock for all HuggingFace model loading (FlagModel, AutoModel, etc.).
# accelerate's init_empty_weights() globally monkey-patches
# nn.Module.register_parameter to allocate on the meta device. This is not
# thread-safe: concurrent from_pretrained() calls corrupt each other's models
# with data-less meta tensors.  Every call site that loads a transformer model
# must acquire this lock to serialize construction.
model_load_lock = threading.Lock()

__all__ = [
    "get_logger",
    "set_global_log_level",
    "get_token_count",
    "TraceHistoryEntry",
    "_serialize_message",
    "model_load_lock",
]
