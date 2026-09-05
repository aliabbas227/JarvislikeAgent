"""
llm_client.py — DeepSeek backend wrapper for Phase 4.

Kept deliberately thin and separate from phaseOne/jarvis.py's call_llm().
If you later want Phase 4 to reuse that seam instead of its own client,
swap build_client()/call_llm_with_tools() for calls into phaseOne — the
tool-calling loop in agent.py doesn't care where the client comes from,
it just needs something with this same call_llm_with_tools() signature.

DeepSeek's API is OpenAI-compatible, so we use the `openai` SDK pointed
at DeepSeek's base_url rather than pulling in a separate SDK.
"""

import os
from openai import OpenAI


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def build_client() -> OpenAI:
    """
    Construct an OpenAI-compatible client pointed at DeepSeek.

    Reads DEEPSEEK_API_KEY (required) and DEEPSEEK_BASE_URL (optional,
    defaults to DeepSeek's standard endpoint) from the environment.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Copy .env.example to .env and add your key."
        )
    base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm_with_tools(client: OpenAI, messages: list, tools: list, model: str | None = None):
    """
    Single call to DeepSeek with tool schemas attached.

    Returns the raw response object. Caller is responsible for checking
    response.choices[0].message.tool_calls to decide whether to execute
    tools or treat the response as a final answer.

    Kept as a single-purpose function (no looping, no tool execution)
    so it's trivially mockable in tests — same pattern used for
    FakeModel in Phase 3.
    """
    model = model or os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
