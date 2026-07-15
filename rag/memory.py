import json
import time
from pathlib import Path
from config import MAX_HISTORY, DATA_DIR

HISTORY_FILE = DATA_DIR / "chat_history.json"


class ConversationMemory:
    def __init__(self, max_history: int = MAX_HISTORY):
        self.max_history = max_history
        self.sessions: dict[str, list[dict]] = {}
        self._load()

    def _load(self):
        if HISTORY_FILE.exists():
            try:
                self.sessions = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.sessions = {}

    def _save(self):
        HISTORY_FILE.write_text(
            json.dumps(self.sessions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_message(self, session_id: str, role: str, content: str, sources: list | None = None):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        msg = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if sources:
            msg["sources"] = sources

        self.sessions[session_id].append(msg)

        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history * 2:]

        self._save()

    def get_history(self, session_id: str) -> list[dict]:
        return self.sessions.get(session_id, [])

    def get_context_messages(self, session_id: str, n: int = 6) -> list[dict]:
        history = self.get_history(session_id)
        recent = history[-n:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear_session(self, session_id: str):
        self.sessions.pop(session_id, None)
        self._save()

    def get_all_sessions(self) -> list[dict]:
        result = []
        for sid, messages in self.sessions.items():
            if messages:
                result.append({
                    "session_id": sid,
                    "message_count": len(messages),
                    "last_message": messages[-1]["content"][:100],
                    "last_timestamp": messages[-1]["timestamp"],
                })
        return sorted(result, key=lambda x: x["last_timestamp"], reverse=True)


memory = ConversationMemory()
