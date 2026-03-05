#!/usr/bin/env python3
"""Render agent JSONL persistence logs into a human-readable timeline.

Usage:
  python agents/utils/render_jsonl_timeline.py .agent_logs/foo.jsonl
  python agents/utils/render_jsonl_timeline.py .agent_logs/foo.jsonl -o timeline.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _block_summary(block: Any, max_chars: int) -> str:
    if isinstance(block, str):
        return _truncate(_collapse_ws(block), max_chars)

    if isinstance(block, dict):
        block_type = block.get("type", "unknown")
        if block_type == "text":
            return f"text: {_truncate(_collapse_ws(str(block.get('text', ''))), max_chars)}"
        if block_type == "tool_use":
            tool_name = block.get("name", "unknown")
            tool_input = _truncate(_collapse_ws(json.dumps(block.get("input", {}), ensure_ascii=False)), max_chars)
            return f"tool_use {tool_name}: {tool_input}"
        if block_type == "tool_result":
            tool_id = block.get("tool_use_id", "unknown")
            content = _truncate(_collapse_ws(str(block.get("content", ""))), max_chars)
            return f"tool_result {tool_id}: {content}"
        return _truncate(_collapse_ws(json.dumps(block, ensure_ascii=False)), max_chars)

    return _truncate(_collapse_ws(str(block)), max_chars)


def _message_summary(message: Any, max_chars: int) -> str:
    if not isinstance(message, dict):
        return _truncate(_collapse_ws(str(message)), max_chars)

    role = message.get("role", "unknown")
    content = message.get("content")

    if isinstance(content, str):
        return f"[{role}] {_truncate(_collapse_ws(content), max_chars)}"

    if isinstance(content, list):
        if not content:
            return f"[{role}] (empty content list)"
        first = _block_summary(content[0], max_chars)
        extra = "" if len(content) == 1 else f" (+{len(content) - 1} blocks)"
        return f"[{role}] {first}{extra}"

    return f"[{role}] {_truncate(_collapse_ws(str(content)), max_chars)}"


def load_events(jsonl_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx, raw in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {idx}: {exc}") from exc
        if not isinstance(evt, dict):
            raise ValueError(f"Line {idx} is not a JSON object")
        events.append(evt)
    return events


def render_timeline(events: list[dict[str, Any]], max_chars: int) -> str:
    if not events:
        return "No events found."

    out: list[str] = []
    prev_messages: list[Any] = []

    for i, evt in enumerate(events, start=1):
        ts = str(evt.get("updated_at", "unknown-time"))
        note = str(evt.get("note", ""))
        session = str(evt.get("session_id", ""))
        messages = evt.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        out.append(f"Event {i}")
        out.append(f"  time: {ts}")
        out.append(f"  note: {note}")
        if session:
            out.append(f"  session: {session}")

        grew_as_prefix = len(messages) >= len(prev_messages) and messages[: len(prev_messages)] == prev_messages
        if grew_as_prefix:
            added = messages[len(prev_messages) :]
            out.append(f"  messages: {len(messages)} (delta +{len(added)})")
            if added:
                out.append("  new:")
                for m in added:
                    out.append(f"    - {_message_summary(m, max_chars)}")
            else:
                out.append("  new: (none)")
        else:
            # This usually means compaction/rewrite replaced conversation history.
            out.append(f"  messages: {len(messages)} (history replaced)")
            if messages:
                out.append("  snapshot:")
                preview = messages[-2:] if len(messages) > 2 else messages
                for m in preview:
                    out.append(f"    - {_message_summary(m, max_chars)}")
            else:
                out.append("  snapshot: (empty)")

        out.append("")
        prev_messages = messages

    return "\n".join(out).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render .agent_logs JSONL file into a readable timeline.")
    parser.add_argument("jsonl", type=Path, help="Path to the .jsonl events file")
    parser.add_argument("-o", "--output", type=Path, help="Optional output file path")
    parser.add_argument("--max-chars", type=int, default=200, help="Max chars for each message preview")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jsonl_path: Path = args.jsonl

    if not jsonl_path.exists():
        print(f"Error: file not found: {jsonl_path}")
        return 1

    if jsonl_path.suffix.lower() != ".jsonl":
        print(f"Warning: input does not end with .jsonl: {jsonl_path}")

    try:
        events = load_events(jsonl_path)
        rendered = render_timeline(events, max_chars=max(40, args.max_chars))
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote timeline: {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
