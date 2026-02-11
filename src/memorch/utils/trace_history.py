from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TraceHistoryEntry:
    """A single entry in the trace history buffer.

    Captures the memory-processed messages for each LLM call within a session.
    """
    
    step: int  # Call number within session (1-indexed)
    input_token_count: int
    compressed_token_count: int
    compression_ratio: float
    memory_method: str
    compressed_messages: List[Dict]  # The actual processed messages sent to LLM

def _serialize_message(msg: Any) -> Dict:
    """Convert a message to a JSON-serializable dictionary.

    Handles raw dicts, pydantic models, and OpenAI SDK objects like
    ChatCompletionMessageToolCall.
    """
    if isinstance(msg, dict):
        # Recursively serialize nested objects
        result = {}
        for key, value in msg.items():
            if hasattr(value, "model_dump"):
                result[key] = value.model_dump()
            elif isinstance(value, list):
                result[key] = [
                    v.model_dump() if hasattr(v, "model_dump") else v for v in value
                ]
            else:
                result[key] = value
        return result
    elif hasattr(msg, "model_dump"):
        return msg.model_dump()
    else:
        return msg
