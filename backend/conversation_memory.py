from collections import defaultdict
from datetime import datetime


# In-memory store: session_id → list of messages
# For production, replace with Redis or a database
_sessions: dict[str, list[dict]] = defaultdict(list)

MAX_HISTORY = 20  # max messages to retain per session


def add_message(session_id: str, role: str, content: str, metadata: dict = None):
    """
    Append a message to the session history.
    role: "user" | "assistant"
    """
    _sessions[session_id].append({
        "role":      role,
        "content":   content,
        "timestamp": datetime.utcnow().isoformat(),
        "metadata":  metadata or {}
    })

    # Keep only the last MAX_HISTORY messages
    if len(_sessions[session_id]) > MAX_HISTORY:
        _sessions[session_id] = _sessions[session_id][-MAX_HISTORY:]


def get_history(session_id: str) -> list[dict]:
    """Return full message history for a session."""
    return _sessions.get(session_id, [])


def get_history_as_text(session_id: str, last_n: int = 6) -> str:
    """
    Return the last N messages as a formatted string for injecting
    into LLM prompts as context.
    """
    history = _sessions.get(session_id, [])[-last_n:]

    if not history:
        return "No prior conversation."

    lines = []
    for msg in history:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {msg['content']}")

    return "\n".join(lines)


def clear_session(session_id: str):
    """Clear all history for a session."""
    if session_id in _sessions:
        del _sessions[session_id]


def list_sessions() -> list[str]:
    """Return all active session IDs."""
    return list(_sessions.keys())