"""Custom exceptions for memory orchestrator."""


class LoopDetectedError(Exception):
    """Raised when an infinite loop is detected in the conversation history.

    This exception is raised when the tail of the conversation contains
    repeating patterns, indicating a potential infinite loop that could
    consume context indefinitely.
    """

    pass
