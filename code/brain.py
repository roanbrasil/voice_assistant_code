"""The 'brain': a tool-calling loop with two interchangeable backends.
BRAIN=claude  -> Claude via the Messages API (needs ANTHROPIC_API_KEY)
BRAIN=ollama  -> a local model via Ollama's OpenAI-compatible endpoint"""
import os

from tools import SCHEMAS, run_tool

SYSTEM = ("You are the household's personal assistant. Answer in English, "
          "in two short sentences at most, because the answer will be spoken aloud. "
          "Use the tools whenever you need data; never make up a temperature, a time "
          "or a device state. Confirm before any action that cannot be undone.")

BRAIN = os.getenv("BRAIN", "claude")
MAX_STEPS = 6  # hard cap on loop iterations: stops the agent spinning in place


class ClaudeBrain:
    def __init__(self, model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model
        self.tools = [{"name": s["name"], "description": s["description"],
                       "input_schema": s["schema"]} for s in SCHEMAS]

    def ask(self, history: list) -> str:
        msgs = list(history)
        for _ in range(MAX_STEPS):
            r = self.client.messages.create(model=self.model, max_tokens=400, system=SYSTEM,
                                            tools=self.tools, messages=msgs)
            msgs.append({"role": "assistant", "content": r.content})
            if r.stop_reason != "tool_use":
                return "".join(b.text for b in r.content if b.type == "text")
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)}
                for b in r.content if b.type == "tool_use"]})
        return "I couldn't finish that right now."


class OllamaBrain:
    def __init__(self, model=os.getenv("OLLAMA_MODEL", "qwen3:4b")):
        from openai import OpenAI
        self.client = OpenAI(base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
                             api_key="ollama")
        self.model = model
        self.tools = [{"type": "function", "function": {
            "name": s["name"], "description": s["description"], "parameters": s["schema"]}}
            for s in SCHEMAS]

    def ask(self, history: list) -> str:
        import json
        msgs = [{"role": "system", "content": SYSTEM}] + list(history)
        for _ in range(MAX_STEPS):
            m = self.client.chat.completions.create(model=self.model, messages=msgs,
                                                    tools=self.tools).choices[0].message
            msgs.append(m.model_dump(exclude_none=True))
            if not m.tool_calls:
                return m.content or ""
            for c in m.tool_calls:
                msgs.append({"role": "tool", "tool_call_id": c.id,
                             "content": run_tool(c.function.name, json.loads(c.function.arguments or "{}"))})
        return "I couldn't finish that right now."


def make_brain():
    return ClaudeBrain() if BRAIN == "claude" else OllamaBrain()
