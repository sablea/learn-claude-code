import json
from datetime import datetime
from pathlib import Path
from typing import Any


class MessagePersister:
    def __init__(self, workdir: Path, file_prefix: str = "s03_todo_write_messages", system: str = ""):
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = workdir / ".agent_logs"
        self.message_log_path = self.log_dir / f"{file_prefix}_{self.session_id}.json"
        self.system = system
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def persist(self, messages: list, note: str = "") -> None:
        try:
            payload = {
                "session_id": self.session_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "note": note,
                "system": self.system,
                "messages": self._to_jsonable(messages),
            }
            self.message_log_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
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
