"""
Multi-turn reply handler.

Handles auto-reply detection, intent classification, hostile exit,
and action-mode switching for the /v1/reply endpoint.
"""

from __future__ import annotations
import logging
from typing import Optional

import llm_client
import state
from prompts.system import REPLY_SYSTEM_PROMPT
from prompts.trigger_templates import build_reply_prompt
from validators import validate_reply_output, parse_llm_json

logger = logging.getLogger(__name__)

# Known auto-reply canned phrases (case-insensitive partial matches)
AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "thanks for reaching out",
    "our team will respond",
    "we will get back to you",
    "automated assistant",
    "automated reply",
    "auto-reply",
    "autoreply",
    "i am an automated",
    "this is an automated",
    "we have received your message",
    "your message has been received",
    "aapki jaankari ke liye",
    "hamari team tak pahuncha",
    "hum aapko jald hi reply",
]

# Hostile / not-interested signals
NOT_INTERESTED_PHRASES = [
    "stop",
    "not interested",
    "don't message",
    "do not message",
    "spam",
    "unsubscribe",
    "stop messaging",
    "leave me alone",
    "nahi chahiye",
    "mat bhejo",
    "band karo",
    "useless",
    "waste of time",
]


def handle_reply(conversation_id: str, merchant_id: str,
                 customer_id: str | None, from_role: str,
                 message: str, turn_number: int) -> dict:
    """
    Handle an incoming merchant/customer reply and produce the bot's next action.
    
    Returns dict with keys: action (send|wait|end), body?, cta?, wait_seconds?, rationale
    """
    # Append incoming message to conversation history
    state.append_conversation(conversation_id, from_role, message)
    conversation = state.get_conversation(conversation_id)

    # ─── Phase 1: Auto-reply detection (fast, no LLM) ─────────────────────
    auto_reply_result = _detect_auto_reply(message, conversation)
    if auto_reply_result is not None:
        return auto_reply_result

    # ─── Phase 2: Hostile / not-interested detection (fast, no LLM) ────────
    hostile_result = _detect_hostile(message)
    if hostile_result is not None:
        return hostile_result

    # ─── Phase 3: LLM-powered reply composition ───────────────────────────
    merchant = state.get_merchant(merchant_id)
    category = None
    if merchant:
        category = state.get_category(merchant.get("category_slug", ""))

    try:
        user_prompt = build_reply_prompt(
            conversation=conversation,
            merchant_message=message,
            merchant=merchant,
            category=category,
        )

        raw_response = llm_client.complete(
            prompt=user_prompt,
            system=REPLY_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=400,
        )

        output = parse_llm_json(raw_response)
        if output is None:
            logger.warning("Failed to parse reply JSON from LLM")
            return _fallback_reply(message, conversation, merchant)

        # Validate
        output, warnings = validate_reply_output(output, conversation)
        if warnings:
            logger.info(f"Reply validation warnings: {warnings}")

        # Track our response in conversation
        if output.get("action") == "send" and output.get("body"):
            state.append_conversation(conversation_id, "vera", output["body"])

        return output

    except Exception as e:
        logger.error(f"Reply LLM call failed: {e}")
        return _fallback_reply(message, conversation, merchant)


def _detect_auto_reply(message: str, conversation: list[dict]) -> Optional[dict]:
    """
    Detect auto-reply patterns. Returns a response dict or None.
    
    Detection rules:
    1. Message matches a known canned phrase
    2. Same message body appears 2+ times in conversation history
    """
    msg_lower = message.lower().strip()

    # Check known canned phrases
    is_canned = any(phrase in msg_lower for phrase in AUTO_REPLY_PHRASES)

    # Check for repeated message from merchant
    merchant_msgs = [t["body"].lower().strip() for t in conversation
                     if t.get("from") in ("merchant", "customer")]
    repeat_count = merchant_msgs.count(msg_lower)

    if is_canned or repeat_count >= 2:
        # Check if we've already tried the human-redirect
        vera_msgs = [t["body"] for t in conversation if t.get("from") == "vera"]
        already_redirected = any("owner" in m.lower() or "manager" in m.lower() or
                                  "directly" in m.lower() for m in vera_msgs)

        if already_redirected or repeat_count >= 2:
            # Second auto-reply or already tried redirect — end gracefully
            return {
                "action": "end",
                "rationale": "Auto-reply detected (repeated canned message). Exiting gracefully after redirect attempt."
            }
        else:
            # First auto-reply — try human redirect
            return {
                "action": "send",
                "body": "Samajh gayi — yeh automated reply lag raha hai. Kya owner/manager se directly baat ho sakti hai? Main unke liye sab ready rakhungi. 🙂",
                "cta": "open_ended",
                "rationale": "First auto-reply detected. Attempting one human-redirect before ending."
            }

    return None


def _detect_hostile(message: str) -> Optional[dict]:
    """Detect hostile or not-interested signals. Returns a response dict or None."""
    msg_lower = message.lower().strip()

    if any(phrase in msg_lower for phrase in NOT_INTERESTED_PHRASES):
        return {
            "action": "end",
            "rationale": "Merchant signaled not interested or hostile. Exiting gracefully."
        }

    return None


def _fallback_reply(message: str, conversation: list[dict],
                     merchant: dict | None) -> dict:
    """Deterministic fallback when LLM is unavailable for reply."""
    msg_lower = message.lower().strip()

    # Simple intent detection
    commitment_words = ["yes", "ok", "let's do it", "go ahead", "proceed", "haan", "chalega", "done", "sure"]
    question_words = ["?", "kya", "how", "what", "when", "why", "kaise", "kitna"]

    if any(w in msg_lower for w in commitment_words):
        merchant_name = ""
        if merchant:
            merchant_name = merchant.get("identity", {}).get("name", "")
        return {
            "action": "send",
            "body": f"Done! I'll set this up for {merchant_name or 'you'} right away. You'll see the updates within 24 hours.",
            "cta": "none",
            "rationale": "Fallback: detected commitment, switching to action mode"
        }
    elif any(w in msg_lower for w in question_words):
        return {
            "action": "send",
            "body": "Good question! Let me check the details and get back to you shortly.",
            "cta": "none",
            "rationale": "Fallback: detected question, providing acknowledgment"
        }
    elif len(conversation) > 6:
        return {
            "action": "end",
            "rationale": "Fallback: conversation has gone 6+ turns, ending gracefully"
        }
    else:
        return {
            "action": "send",
            "body": "Got it. Want me to help with anything specific for your listing?",
            "cta": "open_ended",
            "rationale": "Fallback: generic continuation"
        }
