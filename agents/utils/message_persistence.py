import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .render_jsonl_timeline import load_events, render_timeline
except Exception:  # pragma: no cover - fallback for direct execution contexts
    from render_jsonl_timeline import load_events, render_timeline


class MessagePersister:
    def __init__(self, workdir: Path, file_prefix: str = "s03_todo_write_messages", system: str = ""):
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = workdir / ".agent_logs"
        self.message_log_path = self.log_dir / f"{file_prefix}_{self.session_id}.json"
        self.message_events_path = self.log_dir / f"{file_prefix}_{self.session_id}.jsonl"
        self.message_timeline_path = self.log_dir / f"{file_prefix}_{self.session_id}.timeline.txt"
        self.system = system
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def persist(self, messages: list, note: str = "") -> None:
        try:
            normalized_messages = self._to_jsonable(messages)
            payload = {
                "session_id": self.session_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "note": note,
                "system": self.system,
                "messages": normalized_messages,
            }
            self.message_log_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Keep an append-only timeline so earlier states are not lost.
            with self.message_events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            # Refresh a readable timeline after each JSONL append.
            events = load_events(self.message_events_path)
            timeline_text = render_timeline(events, max_chars=200)
            self.message_timeline_path.write_text(timeline_text, encoding="utf-8")
        except Exception as e:
            print(f"[persist warning] {e}")

    def _to_jsonable(self, value: Any):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        if hasattr(value, "model_dump"):
            try:
                return self._to_jsonable(value.model_dump())
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return self._to_jsonable(value.dict())
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return self._to_jsonable(vars(value))
            except Exception:
                pass
        return str(value)
