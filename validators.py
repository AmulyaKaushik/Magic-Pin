"""
Post-LLM output validation.

Catches taboo words, hallucinations, CTA shape errors, and anti-repetition violations.
"""

from __future__ import annotations
import json
import re
from typing import Optional


def validate_compose_output(output: dict, category: dict, merchant: dict,
                             trigger: dict, customer: dict | None = None) -> tuple[dict, list[str]]:
    """
    Validate and clean a composed message.
    Returns (cleaned_output, list_of_warnings).
    """
    warnings = []

    # Ensure required keys exist
    body = output.get("body", "")
    if not body or not body.strip():
        warnings.append("Empty body")

    # Validate CTA
    cta = output.get("cta", "open_ended")
    valid_ctas = {"open_ended", "binary_yes_stop", "none"}
    if cta not in valid_ctas:
        output["cta"] = "open_ended"
        warnings.append(f"Invalid CTA '{cta}' — defaulted to open_ended")

    # Validate send_as
    send_as = output.get("send_as", "vera")
    if customer and send_as != "merchant_on_behalf":
        output["send_as"] = "merchant_on_behalf"
        warnings.append("Customer present but send_as was not merchant_on_behalf — fixed")
    elif not customer and send_as == "merchant_on_behalf":
        output["send_as"] = "vera"
        warnings.append("No customer but send_as was merchant_on_behalf — fixed to vera")

    # Check taboo words
    taboos = category.get("voice", {}).get("vocab_taboo", [])
    body_lower = body.lower()
    for taboo in taboos:
        if taboo.lower() in body_lower:
            warnings.append(f"Taboo word detected: '{taboo}'")

    # Check suppression key
    if not output.get("suppression_key"):
        output["suppression_key"] = trigger.get("suppression_key", "")

    # Ensure rationale exists
    if not output.get("rationale"):
        output["rationale"] = f"Composed from {trigger.get('kind', 'unknown')} trigger for {merchant.get('identity', {}).get('name', 'merchant')}"

    # Check anti-repetition against conversation history
    conv_history = merchant.get("conversation_history", [])
    for turn in conv_history:
        if turn.get("from") == "vera" and turn.get("body", "").strip() == body.strip():
            warnings.append("ANTI-REPETITION: Body is identical to a previous Vera message")
            break

    return output, warnings


def validate_reply_output(output: dict, conversation: list[dict]) -> tuple[dict, list[str]]:
    """
    Validate a reply handler output.
    Returns (cleaned_output, list_of_warnings).
    """
    warnings = []

    action = output.get("action", "send")
    valid_actions = {"send", "wait", "end"}
    if action not in valid_actions:
        output["action"] = "send"
        warnings.append(f"Invalid action '{action}' — defaulted to send")

    if action == "send":
        body = output.get("body", "")
        if not body or not body.strip():
            warnings.append("Empty body in send action")

        # Anti-repetition check
        for turn in conversation:
            if turn.get("from") == "vera" and turn.get("body", "").strip() == body.strip():
                warnings.append("ANTI-REPETITION: Body identical to previous Vera message")
                break

    if action == "wait":
        wait_s = output.get("wait_seconds", 1800)
        if not isinstance(wait_s, (int, float)) or wait_s < 0:
            output["wait_seconds"] = 1800
            warnings.append("Invalid wait_seconds — defaulted to 1800")

    if not output.get("rationale"):
        output["rationale"] = f"Action: {action}"

    return output, warnings


def parse_llm_json(raw_text: str) -> Optional[dict]:
    """
    Extract a JSON object from LLM output.
    Handles markdown code fences, extra text, etc.
    """
    # Try direct parse first
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block
    brace_match = re.search(r'\{[\s\S]*\}', raw_text)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return None
