#!/usr/bin/env python3
"""Generate a Gemma-3 IT chat template with the unified math few-shot prompt baked in.

The RL runs must use the *exact* prompt that reproduced the eval numbers (GSM8K 37.4 / MATH 24.4
on 4B PT). That prompt = the 12 interleaved MATH+GSM8K exemplars (`fewshot_as_multiturn`) + the
IT chat template. Baking the exemplars into the chat template means one source of truth applied
identically to training rollouts AND validation (verl applies `custom_chat_template` to both).

The exemplars are imported from the eval's own module so the RL prompt can never drift from the
eval prompt. Output: `gemma3_it_fewshot_math.jinja` next to the base IT template.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "rl-distill-scripts" / "lm_eval_tasks"))

from gemma_unified_math_utils import list_fewshot_samples  # noqa: E402

# The message-loop + generation-prompt logic, copied verbatim from
# gemma3_it_chat_template.jinja (Gemma 3 IT format: <start_of_turn>{role}\n...<end_of_turn>).
_MESSAGE_LOOP = """{%- for message in loop_messages -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>\\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\\n'}}
{%- endif -%}
"""

# System-message handling (unused by the math data, but preserved for parity with base template).
_HEADER = """{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\\n\\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\\n\\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
"""


def _fewshot_block() -> str:
    """Render the 12 exemplars as literal Gemma turns (user=question, model=solution)."""
    parts = []
    for ex in list_fewshot_samples():
        parts.append(f"<start_of_turn>user\n{ex['question']}<end_of_turn>\n")
        parts.append(f"<start_of_turn>model\n{ex['solution']}<end_of_turn>\n")
    return "".join(parts)


def main() -> None:
    out = HERE / "gemma3_it_fewshot_math.jinja"
    fewshot = _fewshot_block()
    assert "{% endraw %}" not in fewshot and "{%- endraw -%}" not in fewshot
    # bos, then the static few-shot block (wrapped in {% raw %} so LaTeX braces/$ are literal),
    # then the normal system/message loop + generation prompt.
    template = "{{ bos_token }}" + "{% raw %}" + fewshot + "{% endraw %}" + _HEADER + _MESSAGE_LOOP
    out.write_text(template)
    print(f"wrote {out}  ({len(list_fewshot_samples())} shots, {len(template)} chars)")


if __name__ == "__main__":
    main()
