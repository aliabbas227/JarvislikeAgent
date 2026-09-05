"""
agent.py — Phase 4 entry point: text-only CLI agent with tool calling.

Deliberately no voice, no wake word, no memory — those are Phases 2, 3,
and 5, and standalone-phase discipline says this doesn't reach into
them. Just: type a message, watch the LLM decide whether to call a
tool, see the result folded into its reply.

Backend: DeepSeek via the OpenAI-compatible SDK (see llm_client.py).
"""

import json
import os
from dotenv import load_dotenv

from llm_client import build_client, call_llm_with_tools
from tools import TOOL_SCHEMAS, execute_tool

MAX_TOOL_HOPS = 4  # safety cap so a confused model can't loop forever


def run_turn(client, messages: list) -> str:
    """
    Run one user turn to completion: call the LLM, execute any tool
    calls it requests, feed results back, repeat until the model
    returns a plain text answer (or MAX_TOOL_HOPS is hit).

    Mutates `messages` in place so conversation history accumulates
    correctly across turns.
    """
    for _ in range(MAX_TOOL_HOPS):
        response = call_llm_with_tools(client, messages, TOOL_SCHEMAS)
        message = response.choices[0].message

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            # Final answer — no more tools requested.
            messages.append({"role": "assistant", "content": message.content})
            return message.content

        # The assistant message that contains the tool call requests
        # must be appended before the tool results, per the API's
        # expected message ordering.
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            }
        )

        for tc in tool_calls:
            name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            try:
                result = execute_tool(name, arguments)
            except (KeyError, TypeError) as e:
                # Surface the error back to the model rather than
                # crashing the whole turn — let it decide how to recover
                # or apologize to the user.
                result = {"error": str(e)}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    return "(gave up after too many tool calls in a row — check MAX_TOOL_HOPS)"


def main():
    load_dotenv()
    client = build_client()

    system_prompt = (
        "You are Jarvis, a helpful assistant with access to tools. "
        "Use tools when they'd give a better answer than guessing. "
        "Keep replies concise."
    )
    messages = [{"role": "system", "content": system_prompt}]

    print("Jarvis Phase 4 agent (DeepSeek backend). Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nJarvis: Shutting down. Goodbye.")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Jarvis: Goodbye.")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        reply = run_turn(client, messages)
        print(f"Jarvis: {reply}\n")


if __name__ == "__main__":
    main()