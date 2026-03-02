# ACE Strategy — Context & Output per Step

*2026-03-02T14:18:58Z by Showboat 0.6.1*
<!-- showboat-id: 4d06b2c9-13d5-412c-90d5-3519950b76b6 -->

This document traces what each ACE sub-agent (Reflector, Curator, Generator) receives as input and what it returns, across two consecutive steps of apply_ace_strategy(). No real LLM calls are made — the LLM responses are stubbed so you can see the exact data flow without network dependencies.

## The three sub-agents and their roles

| Agent | When it runs | What it receives | What it produces |
|---|---|---|---|
| **Reflector** | Step 2+ (skipped on step 1) | task, *previous* reasoning trace + predicted answer, bullets that were used last step, env feedback | reflection text, bullet tags (helpful/harmful/neutral) |
| **Curator** | Every N steps (curator_frequency, default 1) | current playbook, stats, reflection from this step, question context, step number, token budget | updated playbook, next_global_id, list of ADD/REMOVE/UPDATE ops |
| **Generator** | Every step | task, current playbook, reflection from this step, last 3 messages as context | reasoning trace JSON, bullet IDs used |

The Reflector feeds into the Curator (reflection text), and the Generator feeds into the *next* step's Reflector (reasoning trace). Each agent calls generate_plain() — bypassing all memory processing — with a single user message containing its rendered prompt.

## Step 1 — first call to apply_ace_strategy()

On the very first step the ACE state is empty: no previous reasoning trace, no bullets. This means:
- **Reflector** is **skipped** (requires last_reasoning_trace to be set)
- **Curator** runs to bootstrap the empty playbook
- **Generator** runs and stores its output in state for the next step's Reflector

```bash
uv run python docs/_ace_trace.py 2>/dev/null
```

```output

########################################################################
# STEP 1  (first call — Reflector skipped, no prior trace)
########################################################################

========================================================================
INPUT TO apply_ace_strategy()
========================================================================
  [0] role='user'        'What is the price of Widget X?'
  [1] role='system'      '<haystack>unrelated filler content. unrelated filler content. unrelated filler content. unrelated filler content. unrela'...
2026-03-02 15:21:06,394 - INFO - ACEStrategy - 
ACE Strategy - Step 1
2026-03-02 15:21:06,395 - INFO - ACEStrategy - ============================================================
2026-03-02 15:21:06,395 - INFO - ACEStrategy - State: last_reasoning_trace=<empty>
2026-03-02 15:21:06,395 - INFO - ACEStrategy - State: last_bullet_ids=[]
2026-03-02 15:21:06,395 - INFO - ACEStrategy - State: last_reflection=<empty>
2026-03-02 15:21:06,395 - INFO - ACEStrategy - State: next_global_id=1
2026-03-02 15:21:06,395 - INFO - ACEStrategy - Config: curator_frequency=1
2026-03-02 15:21:06,395 - INFO - ACEStrategy - Playbook preview (first 200 chars): # Agent Playbook

## Task Decomposition (TSD)
<!-- Break down complex tasks into manageable steps -->

## Error Handling (ERR)
<!-- Strategies for detecting and recovering from errors -->

## Context ...
2026-03-02 15:21:06,395 - INFO - ACEStrategy - Reflector conditions: has_reasoning=False, has_bullets=False

--- CURATOR (step 1) ---
  Input fields:
    current_playbook : <EMPTY_PLAYBOOK_TEMPLATE>
    recent_reflection: (empty — no prior reflection)
    question_context : 'No additional context'
    step             : 1
    token_budget     : 4096
  Prompt sent (first 400 chars):
    # Curator Agent Prompt  You are a playbook curator that maintains and
    improves the agent's knowledge base.  ## Your Task Review the current
    playbook and recent performance, then decide what operations to perform: -
    ADD: Insert new bullet with learned insight - REMOVE: Delete outdated or
    harmful bullet - UPDATE: Modify existing bullet content - NONE: No changes
    needed  ## Current Playbook # Agent P
  LLM response:
    {"reasoning": "Empty playbook \u2014 add a foundational tool-usage bullet.", "operations": [{"op": "ADD", "section": "tool_usage", "content": "Always call a tool before answering; never guess."}]}
  Playbook after curation (non-empty lines):
    ## Tool Usage (TLS)
    <!-- Best practices for using available tools -->
    [1] helpful=0 harmful=0 :: Always call a tool before answering; never guess.
    ## Communication (COM)
    <!-- Guidelines for clear and effective responses -->

--- GENERATOR (step 1) ---
  Input fields:
    question : 'What is the price of Widget X?'
    playbook : <playbook after curator, see above>
    context  : last 3 msgs joined (role: content)
    reflection: (empty — no prior reflection)
  Prompt sent (first 400 chars):
    # Generator Agent Prompt  You are an intelligent reasoning agent that uses a
    living playbook to guide your decision-making process.  ## Your Task Analyze
    the given question and context, then generate a response that includes: 1.
    Your reasoning trace (step-by-step thought process) 2. The next action to
    take — a tool call recommendation, OR a final answer only when all required
    information is alread
  LLM response:
    {"reasoning_trace": "Task needs a live lookup. I will call search_product with name=Widget X.", "response": "Call search_product with name='Widget X' to retrieve the current price.", "bullet_ids_used": [1]}
  Stored in state:
    last_reasoning_trace: '{"reasoning_trace": "Task needs a live lookup. I will call search_product with name=Widget X.", "res'...
    last_bullet_ids     : [1]

========================================================================
OUTPUT of apply_ace_strategy() → sent to main LLM
========================================================================
  [0] role='system'      '## PLAYBOOK  # Agent Playbook  ## Task Decomposition (TSD) <!-- Break down complex tasks into manageable steps -->  ## E'...
  [1] role='user'        'What is the price of Widget X?'
  [2] role='system'      '<haystack>unrelated filler content. unrelated filler content. unrelated filler content. unrelated filler content. unrela'...
  [3] role='system'      '## ACE REASONING TRACE (Step 1)  ### Reasoning {"reasoning_trace": "Task needs a live lookup. I will call search_product'...

  Message count: 2 → 4  (+2 ACE system messages prepended/appended)


########################################################################
# STEP 2  (second call — Reflector NOW runs on step-1 trace)
########################################################################

========================================================================
INPUT TO apply_ace_strategy()  (result_s1 + tool round)
========================================================================
  [0] role='system'      '## PLAYBOOK  # Agent Playbook  ## Task Decomposition (TSD) <!-- Break down complex tasks into manageable steps -->  ## E'...
  [1] role='user'        'What is the price of Widget X?'
  [2] role='system'      '<haystack>unrelated filler content. unrelated filler content. unrelated filler content. unrelated filler content. unrela'...
  [3] role='system'      '## ACE REASONING TRACE (Step 1)  ### Reasoning {"reasoning_trace": "Task needs a live lookup. I will call search_product'...
  [4] role='assistant'   ''
  [5] role='tool'        '{"price": "$42.99", "currency": "USD"}'
2026-03-02 15:21:06,396 - INFO - ACEStrategy - 
ACE Strategy - Step 2
2026-03-02 15:21:06,396 - INFO - ACEStrategy - ============================================================
2026-03-02 15:21:06,397 - INFO - ACEStrategy - State: last_reasoning_trace=<set>
2026-03-02 15:21:06,397 - INFO - ACEStrategy - State: last_bullet_ids=[1]
2026-03-02 15:21:06,397 - INFO - ACEStrategy - State: last_reflection=<empty>
2026-03-02 15:21:06,397 - INFO - ACEStrategy - State: next_global_id=2
2026-03-02 15:21:06,397 - INFO - ACEStrategy - Config: curator_frequency=1
2026-03-02 15:21:06,397 - INFO - ACEStrategy - Playbook preview (first 200 chars): # Agent Playbook

## Task Decomposition (TSD)
<!-- Break down complex tasks into manageable steps -->

## Error Handling (ERR)
<!-- Strategies for detecting and recovering from errors -->

## Context ...
2026-03-02 15:21:06,397 - INFO - ACEStrategy - Reflector conditions: has_reasoning=True, has_bullets=True
2026-03-02 15:21:06,397 - INFO - ACEStrategy - Bullets extracted for reflection: [1] helpful=0 harmful=0 :: Always call a tool before answering; never guess....

--- REFLECTOR (step 2) ---
  Input fields (all from ACEState set during step 1):
    question         : 'What is the price of Widget X?'
    reasoning_trace  : last_reasoning_trace (set by Generator step 1)
    predicted_answer : same as reasoning_trace
    environment_feedback: 'No Feedback'  (hardcoded placeholder)
    bullets_used     : extract_playbook_bullets(playbook, [1])
  Prompt sent (first 400 chars):
    # Reflector Agent Prompt  You are a critical analyzer that evaluates agent
    performance and provides feedback for playbook improvement.  ## Your Task
    Analyze the agent's reasoning and outcome, then provide: 1. What went right
    and what went wrong 2. Which playbook bullets were helpful vs harmful 3.
    Suggestions for playbook improvements  ## Question What is the price of
    Widget X?  ## Reasoning Trace
  LLM response:
    {"reflection": "Bullet 1 correctly directed the agent to use a tool instead of guessing.", "bullet_tags": [{"bullet_id": 1, "tag": "helpful"}], "improvement_suggestions": "None needed yet."}
  Stored in state:
    last_reflection: '{"reflection": "Bullet 1 correctly directed the agent to use a tool instead of guessing.", "bullet_t'...

--- CURATOR (step 2) ---
  Input fields:
    current_playbook : <playbook after step-1 curation>
    recent_reflection: last_reflection (set by Reflector above)
    question_context : 'No additional context'
    step             : 2
  Prompt sent (first 400 chars):
    # Curator Agent Prompt  You are a playbook curator that maintains and
    improves the agent's knowledge base.  ## Your Task Review the current
    playbook and recent performance, then decide what operations to perform: -
    ADD: Insert new bullet with learned insight - REMOVE: Delete outdated or
    harmful bullet - UPDATE: Modify existing bullet content - NONE: No changes
    needed  ## Current Playbook # Agent P
  LLM response:
    {"reasoning": "Bullet 1 is performing well, no changes needed.", "operations": []}
  Operations applied: []  (no changes)

--- GENERATOR (step 2) ---
  Input fields:
    question  : 'What is the price of Widget X?'
    playbook  : <playbook unchanged from step 1>
    context   : last 3 msgs  (ACE system msg + tool call + tool result)
    reflection: last_reflection (set by Reflector above)
  Prompt sent (first 400 chars):
    # Generator Agent Prompt  You are an intelligent reasoning agent that uses a
    living playbook to guide your decision-making process.  ## Your Task Analyze
    the given question and context, then generate a response that includes: 1.
    Your reasoning trace (step-by-step thought process) 2. The next action to
    take — a tool call recommendation, OR a final answer only when all required
    information is alread
  LLM response:
    {"reasoning_trace": "Tool was called last step. Now I need to parse the result and return the price.", "response": "Call format_price with raw_result=<tool_output> to format the price for the user.", "bullet_ids_used": [1]}

========================================================================
OUTPUT of apply_ace_strategy() → sent to main LLM
========================================================================
  [0] role='system'      '## PLAYBOOK  # Agent Playbook  ## Task Decomposition (TSD) <!-- Break down complex tasks into manageable steps -->  ## E'...
  [1] role='system'      '## PLAYBOOK  # Agent Playbook  ## Task Decomposition (TSD) <!-- Break down complex tasks into manageable steps -->  ## E'...
  [2] role='user'        'What is the price of Widget X?'
  [3] role='system'      '<haystack>unrelated filler content. unrelated filler content. unrelated filler content. unrelated filler content. unrela'...
  [4] role='system'      '## ACE REASONING TRACE (Step 1)  ### Reasoning {"reasoning_trace": "Task needs a live lookup. I will call search_product'...
  [5] role='assistant'   ''
  [6] role='tool'        '{"price": "$42.99", "currency": "USD"}'
  [7] role='system'      '## ACE REASONING TRACE (Step 2)  ### Reasoning {"reasoning_trace": "Tool was called last step. Now I need to parse the r'...

  Message count: 6 → 8  (+2 ACE messages, replacing prior pair)

########################################################################
# SUMMARY — what grows and what stays constant
########################################################################

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

```
