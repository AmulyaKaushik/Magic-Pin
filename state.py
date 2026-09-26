"""
In-memory state management for the Vera bot.

All context data lives here for the duration of the test window.
No persistence — purely in-memory, designed for idempotent version-tracked updates.
"""

from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Any, Optional

START_TIME = time.time()

# ─── Primary stores ───────────────────────────────────────────────────────────

# (scope, context_id) → {"version": int, "payload": dict}
contexts: dict[tuple[str, str], dict[str, Any]] = {}

# conversation_id → [{"from": "vera"|"merchant"|"customer", "body": str, "ts": str}]
conversations: dict[str, list[dict[str, Any]]] = {}

# suppression_keys already fired this session
suppression_log: set[str] = set()

# auto-reply count per merchant / conv
auto_reply_counts: dict[str, int] = {}

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


# ─── Context operations ──────────────────────────────────────────────────────

def push_context(scope: str, context_id: str, version: int, payload: dict) -> dict:
    """
    Store or update a context. Returns the API response dict.
    
    - Same version → no-op (idempotent)
    - Higher version → atomic replace
    - Lower version → reject with 409
    """
    if scope not in VALID_SCOPES:
        return {"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {VALID_SCOPES}"}

    key = (scope, context_id)
    current = contexts.get(key)

    if current is not None:
        if current["version"] > version:
            return {"accepted": False, "reason": "stale_version", "current_version": current["version"]}
        if current["version"] == version:
            # Idempotent — same version is a no-op, still report success
            return {"accepted": True, "ack_id": f"ack_{context_id}_v{version}",
                    "stored_at": datetime.now(timezone.utc).isoformat()}

    contexts[key] = {"version": version, "payload": payload}
    return {"accepted": True, "ack_id": f"ack_{context_id}_v{version}",
            "stored_at": datetime.now(timezone.utc).isoformat()}


# ─── Lookup helpers ───────────────────────────────────────────────────────────

def get_context(scope: str, context_id: str) -> Optional[dict]:
    """Get the payload for a stored context, or None."""
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def get_category(slug: str) -> Optional[dict]:
    return get_context("category", slug)


def get_merchant(merchant_id: str) -> Optional[dict]:
    return get_context("merchant", merchant_id)


def get_customer(customer_id: str) -> Optional[dict]:
    return get_context("customer", customer_id)


def get_trigger(trigger_id: str) -> Optional[dict]:
    return get_context("trigger", trigger_id)


# ─── Context counts for healthz ───────────────────────────────────────────────

def get_context_counts() -> dict[str, int]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts


def get_uptime() -> int:
    return int(time.time() - START_TIME)


# ─── Conversation history ─────────────────────────────────────────────────────

def append_conversation(conversation_id: str, from_role: str, body: str, ts: str | None = None):
    """Append a turn to a conversation."""
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    conversations[conversation_id].append({
        "from": from_role,
        "body": body,
        "ts": ts or datetime.now(timezone.utc).isoformat(),
    })


def get_conversation(conversation_id: str) -> list[dict]:
    """Get the full turn history for a conversation."""
    return conversations.get(conversation_id, [])


# ─── Suppression ──────────────────────────────────────────────────────────────

def is_suppressed(suppression_key: str) -> bool:
    return suppression_key in suppression_log


def mark_suppressed(suppression_key: str):
    suppression_log.add(suppression_key)


def record_auto_reply(key: str) -> int:
    """Increment and return the auto-reply count for a merchant or conversation."""
    auto_reply_counts[key] = auto_reply_counts.get(key, 0) + 1
    return auto_reply_counts[key]


def get_auto_reply_count(key: str) -> int:
    return auto_reply_counts.get(key, 0)


def clear_all():
    """Clear all in-memory state (for teardown/testing)."""
    contexts.clear()
    conversations.clear()
    suppression_log.clear()
    auto_reply_counts.clear()

