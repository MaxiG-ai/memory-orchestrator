"""
Traces ACE strategy context and output across two steps.
No real LLM calls — responses are stubbed.
"""

import sys, json, textwrap

sys.path.insert(0, "src")

from unittest.mock import MagicMock
from memorch.strategies.ace.ace_strategy import ACEState, apply_ace_strategy
from memorch.strategies.ace.playbook_utils import EMPTY_PLAYBOOK_TEMPLATE

SEP = "=" * 72


def preview(text, n=120):
    s = (text or "")[:n].replace("\n", " ")
    suffix = "..." if len(text or "") > n else ""
    return repr(s) + suffix


def show_messages(label, msgs):
    print(f"\n{SEP}")
    print(label)
    print(SEP)
    for i, m in enumerate(msgs):
        print(f"  [{i}] role={m['role']!r:12s}  {preview(m.get('content') or '')}")


# ── Stub LLM responses ──────────────────────────────────────────────────────

CURATOR_RESP_S1 = json.dumps(
    {
        "reasoning": "Empty playbook — add a foundational tool-usage bullet.",
        "operations": [
            {
                "op": "ADD",
                "section": "tool_usage",
                "content": "Always call a tool before answering; never guess.",
            }
        ],
    }
)
GENERATOR_RESP_S1 = json.dumps(
    {
        "reasoning_trace": "Task needs a live lookup. I will call search_product with name=Widget X.",
        "response": "Call search_product with name='Widget X' to retrieve the current price.",
        "bullet_ids_used": [1],
    }
)
REFLECTOR_RESP_S2 = json.dumps(
    {
        "reflection": "Bullet 1 correctly directed the agent to use a tool instead of guessing.",
        "bullet_tags": [{"bullet_id": 1, "tag": "helpful"}],
        "improvement_suggestions": "None needed yet.",
    }
)
CURATOR_RESP_S2 = json.dumps(
    {"reasoning": "Bullet 1 is performing well, no changes needed.", "operations": []}
)
GENERATOR_RESP_S2 = json.dumps(
    {
        "reasoning_trace": "Tool was called last step. Now I need to parse the result and return the price.",
        "response": "Call format_price with raw_result=<tool_output> to format the price for the user.",
        "bullet_ids_used": [1],
    }
)


# Map which stub to return based on prompt content
def make_llm(captured, step):
    def fake_generate_plain(input_messages, model=None, **kw):
        prompt = input_messages[0]["content"]
        m = MagicMock()
        m.choices = [MagicMock()]
        if "playbook curator" in prompt.lower():
            captured[f"s{step}_curator_prompt"] = prompt
            m.choices[0].message.content = (
                CURATOR_RESP_S2 if step == 2 else CURATOR_RESP_S1
            )
        elif "critical analyzer" in prompt.lower():
            captured[f"s{step}_reflector_prompt"] = prompt
            m.choices[0].message.content = REFLECTOR_RESP_S2
        else:
            captured[f"s{step}_generator_prompt"] = prompt
            m.choices[0].message.content = (
                GENERATOR_RESP_S2 if step == 2 else GENERATOR_RESP_S1
            )
        return m

    llm = MagicMock()
    llm.generate_plain.side_effect = fake_generate_plain
    return llm


settings = MagicMock(
    curator_frequency=1,
    generator_model="gpt-4-1-mini",
    reflector_model="gpt-4-1-mini",
    curator_model="gpt-4-1-mini",
    playbook_token_budget=4096,
)

# ── Incoming messages ───────────────────────────────────────────────────────
# Simulates a user query + ~1 k of haystack filler
HAYSTACK = "unrelated filler content. " * 40  # ~1 k chars (short for readability)
messages_s1 = [
    {"role": "user", "content": "What is the price of Widget X?"},
    {"role": "system", "content": f"<haystack>{HAYSTACK}</haystack>"},
]

# ── STEP 1 ──────────────────────────────────────────────────────────────────
print(f"\n{'#' * 72}")
print("# STEP 1  (first call — Reflector skipped, no prior trace)")
print(f"{'#' * 72}")

captured = {}
state = ACEState()
llm_s1 = make_llm(captured, step=1)

show_messages("INPUT TO apply_ace_strategy()", messages_s1)

result_s1 = apply_ace_strategy(messages_s1, llm_s1, settings, state)

print(f"\n--- CURATOR (step 1) ---")
print("  Input fields:")
print("    current_playbook : <EMPTY_PLAYBOOK_TEMPLATE>")
print("    recent_reflection: (empty — no prior reflection)")
print("    question_context : 'No additional context'")
print("    step             : 1")
print("    token_budget     : 4096")
print("  Prompt sent (first 400 chars):")
print(
    textwrap.indent(
        textwrap.fill(captured.get("s1_curator_prompt", "<not called>")[:400], 76),
        "    ",
    )
)
print("  LLM response:")
print(textwrap.indent(CURATOR_RESP_S1, "    "))
print("  Playbook after curation (non-empty lines):")
for l in [l for l in state.playbook.splitlines() if l.strip()][-5:]:
    print("   ", l)

print(f"\n--- GENERATOR (step 1) ---")
print("  Input fields:")
print("    question : 'What is the price of Widget X?'")
print("    playbook : <playbook after curator, see above>")
print("    context  : last 3 msgs joined (role: content)")
print("    reflection: (empty — no prior reflection)")
print("  Prompt sent (first 400 chars):")
print(
    textwrap.indent(
        textwrap.fill(captured.get("s1_generator_prompt", "<not called>")[:400], 76),
        "    ",
    )
)
print("  LLM response:")
print(textwrap.indent(GENERATOR_RESP_S1, "    "))
print("  Stored in state:")
print("    last_reasoning_trace:", repr(state.last_reasoning_trace[:100]) + "...")
print("    last_bullet_ids     :", state.last_bullet_ids)

show_messages("OUTPUT of apply_ace_strategy() → sent to main LLM", result_s1)
print(
    f"\n  Message count: {len(messages_s1)} → {len(result_s1)}  (+2 ACE system messages prepended/appended)"
)

# ── Simulate tool call result coming back ───────────────────────────────────
messages_s2 = result_s1 + [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc1",
                "type": "function",
                "function": {
                    "name": "search_product",
                    "arguments": '{"name": "Widget X"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "content": '{"price": "$42.99", "currency": "USD"}',
        "tool_call_id": "tc1",
    },
]

# ── STEP 2 ──────────────────────────────────────────────────────────────────
print(f"\n\n{'#' * 72}")
print("# STEP 2  (second call — Reflector NOW runs on step-1 trace)")
print(f"{'#' * 72}")

llm_s2 = make_llm(captured, step=2)
show_messages("INPUT TO apply_ace_strategy()  (result_s1 + tool round)", messages_s2)

result_s2 = apply_ace_strategy(messages_s2, llm_s2, settings, state)

print(f"\n--- REFLECTOR (step 2) ---")
print("  Input fields (all from ACEState set during step 1):")
print("    question         : 'What is the price of Widget X?'")
print("    reasoning_trace  : last_reasoning_trace (set by Generator step 1)")
print("    predicted_answer : same as reasoning_trace")
print("    environment_feedback: 'No Feedback'  (hardcoded placeholder)")
print("    bullets_used     : extract_playbook_bullets(playbook, [1])")
print("  Prompt sent (first 400 chars):")
print(
    textwrap.indent(
        textwrap.fill(captured.get("s2_reflector_prompt", "<not called>")[:400], 76),
        "    ",
    )
)
print("  LLM response:")
print(textwrap.indent(REFLECTOR_RESP_S2, "    "))
print("  Stored in state:")
print("    last_reflection:", repr(state.last_reflection[:100]) + "...")

print(f"\n--- CURATOR (step 2) ---")
print("  Input fields:")
print("    current_playbook : <playbook after step-1 curation>")
print("    recent_reflection: last_reflection (set by Reflector above)")
print("    question_context : 'No additional context'")
print("    step             : 2")
print("  Prompt sent (first 400 chars):")
print(
    textwrap.indent(
        textwrap.fill(captured.get("s2_curator_prompt", "<not called>")[:400], 76),
        "    ",
    )
)
print("  LLM response:")
print(textwrap.indent(CURATOR_RESP_S2, "    "))
print("  Operations applied: []  (no changes)")

print(f"\n--- GENERATOR (step 2) ---")
print("  Input fields:")
print("    question  : 'What is the price of Widget X?'")
print("    playbook  : <playbook unchanged from step 1>")
print("    context   : last 3 msgs  (ACE system msg + tool call + tool result)")
print("    reflection: last_reflection (set by Reflector above)")
print("  Prompt sent (first 400 chars):")
print(
    textwrap.indent(
        textwrap.fill(captured.get("s2_generator_prompt", "<not called>")[:400], 76),
        "    ",
    )
)
print("  LLM response:")
print(textwrap.indent(GENERATOR_RESP_S2, "    "))

show_messages("OUTPUT of apply_ace_strategy() → sent to main LLM", result_s2)
print(
    f"\n  Message count: {len(messages_s2)} → {len(result_s2)}  (+2 ACE messages, replacing prior pair)"
)

print(f"\n{'#' * 72}")
print("# SUMMARY — what grows and what stays constant")
print(f"{'#' * 72}")
print("""
  Each call to apply_ace_strategy() always:
    1. Prepends  [system] ## PLAYBOOK  (grows as bullets are added)
    2. Passes through ALL original messages unchanged
    3. Appends   [system] ## ACE REASONING TRACE

  What each agent sees — and does NOT see:
    Reflector  : task + prior reasoning trace + prior bullets.  Does NOT see haystack.
    Curator    : playbook + stats + reflection + context str.   Does NOT see haystack.
    Generator  : task + playbook + reflection + last 3 msgs.   Sees PARTIAL haystack
                 only if it falls within the last 3 messages.

  The main LLM sees EVERYTHING: full haystack + playbook + reasoning trace.
  Token count can only grow across steps.
""")
