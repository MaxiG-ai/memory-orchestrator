from __future__ import annotations

from types import SimpleNamespace

from memorch.utils.token_count import _iter_message_text_parts, get_token_count


def test_iter_message_text_parts_collects_content_and_tool_payloads() -> None:
    """Verify extraction includes all token-bearing text fields used by chat payloads.

    This test validates that `_iter_message_text_parts` captures user-visible content,
    tool-call metadata (function name and arguments), and serialized legacy
    `function_call` payloads. It also checks that empty strings and unsupported
    shapes are skipped, so downstream token counting can focus on meaningful text.
    """
    message = {
        "content": "Summarize the report",
        "tool_calls": [
            {"function": {"name": "search", "arguments": "{\"query\":\"Q4\"}"}},
            {"function": {"name": "", "arguments": ""}},
            "invalid-tool-call-entry",
        ],
        "function_call": {"name": "finalize"},
    }

    assert _iter_message_text_parts(message) == [
        "Summarize the report",
        "search",
        '{"query":"Q4"}',
        "{'name': 'finalize'}",
    ]


def test_get_token_count_ignores_non_dict_entries_and_counts_list_messages(monkeypatch) -> None:
    """Ensure list payload counting is robust to mixed item types and sums encodings.

    The production function accepts an arbitrary list-like payload and should only
    process dict messages. This test injects a deterministic fake tokenizer so the
    result reflects the exact number of collected text fragments without depending on
    tiktoken model files or network access.
    """

    monkeypatch.setattr(
        "memorch.utils.token_count.tiktoken.get_encoding",
        lambda _: SimpleNamespace(encode=lambda text: [ord(c) for c in text]),
    )

    count = get_token_count(
        [
            {"content": "ab"},
            "not-a-message",
            {"tool_calls": [{"function": {"name": "x", "arguments": "yz"}}]},
        ]
    )

    assert count == 5


def test_get_token_count_handles_single_message_and_unknown_payload_types(monkeypatch) -> None:
    """Confirm dict payloads are counted and unsupported top-level payloads return zero.

    This test checks both supported and unsupported entry points in one place:
    `get_token_count` should count a single dict message and safely return zero for
    unrelated payload types (e.g., strings), preventing accidental crashes.
    """

    monkeypatch.setattr(
        "memorch.utils.token_count.tiktoken.get_encoding",
        lambda _: SimpleNamespace(encode=lambda text: [text]),
    )

    assert get_token_count({"content": "hello", "function_call": "done"}) == 2
    assert get_token_count("plain string payload") == 0
