import json
import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class TextBlock:
    type: str
    text: str


@dataclass
class ToolUseBlock:
    type: str
    id: str
    name: str
    input: dict


@dataclass
class MessageResponse:
    content: list
    stop_reason: str


class _MessagesAPI:
    def __init__(self, client: OpenAI):
        self._client = client

    def create(self, model: str, messages: list, system: str = None, tools: list = None, max_tokens: int = 8000):
        req_messages = []
        if system:
            req_messages.append({"role": "system", "content": system})

        req_messages.extend(self._convert_messages(messages))
        req_tools = self._convert_tools(tools or [])

        resp = self._client.chat.completions.create(
            model=model,
            messages=req_messages,
            tools=req_tools if req_tools else None,
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message

        blocks = []
        if msg.content:
            blocks.append(TextBlock(type="text", text=msg.content))

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                blocks.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=tc.id,
                        name=tc.function.name,
                        input=args,
                    )
                )

        stop_reason = "tool_use" if msg.tool_calls else "end_turn"
        return MessageResponse(content=blocks, stop_reason=stop_reason)

    def _convert_tools(self, tools: list) -> list:
        converted = []
        for t in tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
            )
        return converted

    def _convert_messages(self, messages: list) -> list:
        converted = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "assistant":
                converted.append(self._assistant_message(content))
                continue

            if role == "user" and isinstance(content, list):
                user_texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "tool_result":
                        converted.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.get("tool_use_id", ""),
                                "content": str(part.get("content", "")),
                            }
                        )
                    elif isinstance(part, dict) and part.get("type") == "text":
                        user_texts.append(str(part.get("text", "")))
                if user_texts:
                    converted.append({"role": "user", "content": "\n".join(user_texts)})
                continue

            if role in ("user", "assistant", "system", "tool"):
                converted.append({"role": role, "content": str(content)})

        return converted

    def _assistant_message(self, content) -> dict:
        if not isinstance(content, list):
            return {"role": "assistant", "content": str(content or "")}

        text_parts = []
        tool_calls = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")

            if block_type == "text":
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text", "")
                text_parts.append(str(text or ""))
            elif block_type == "tool_use":
                block_id = getattr(block, "id", None)
                name = getattr(block, "name", None)
                tool_input = getattr(block, "input", None)
                if isinstance(block, dict):
                    block_id = block.get("id", block_id)
                    name = block.get("name", name)
                    tool_input = block.get("input", tool_input)
                tool_calls.append(
                    {
                        "id": str(block_id or ""),
                        "type": "function",
                        "function": {
                            "name": str(name or ""),
                            "arguments": json.dumps(tool_input or {}),
                        },
                    }
                )

        message = {"role": "assistant", "content": "\n".join(t for t in text_parts if t)}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message


class Anthropic:
    def __init__(self, base_url: str = None):
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
        )
        if not api_key and base_url:
            api_key = "local-dev-key"
        if not api_key:
            api_key = "missing-api-key"
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.messages = _MessagesAPI(self._client)
