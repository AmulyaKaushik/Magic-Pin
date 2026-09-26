"""
Core composer — the compose() function.

Takes 4 contexts (category, merchant, trigger, customer?) and returns
a composed WhatsApp message via a single LLM call with trigger-routed prompting.
"""

from __future__ import annotations
import logging
from typing import Optional

import llm_client
from prompts.system import COMPOSER_SYSTEM_PROMPT
from prompts.trigger_templates import build_compose_prompt
from validators import validate_compose_output, parse_llm_json

logger = logging.getLogger(__name__)


def compose(category: dict, merchant: dict, trigger: dict,
            customer: dict | None = None) -> dict:
    """
    Compose a WhatsApp message from the 4-context framework.
    
    Returns:
        dict with keys: body, cta, send_as, suppression_key, rationale
    """
    # Build the trigger-routed prompt
    user_prompt = build_compose_prompt(category, merchant, trigger, customer)

    # Call LLM
    try:
        raw_response = llm_client.complete(
            prompt=user_prompt,
            system=COMPOSER_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=1500,
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return _fallback_compose(category, merchant, trigger, customer)

    # Parse JSON from LLM output
    output = parse_llm_json(raw_response)
    if output is None:
        logger.warning(f"Failed to parse LLM JSON, attempting retry")
        # One retry with explicit JSON instruction
        try:
            raw_response = llm_client.complete(
                prompt=user_prompt + "\n\nIMPORTANT: Your previous response was not valid JSON. Return ONLY a JSON object with keys: body, cta, send_as, suppression_key, rationale.",
                system=COMPOSER_SYSTEM_PROMPT,
                temperature=0,
                max_tokens=1500,
            )
            output = parse_llm_json(raw_response)
        except Exception as e:
            logger.error(f"Retry LLM call failed: {e}")

    if output is None:
        logger.error("Both LLM attempts failed to produce valid JSON — using fallback")
        return _fallback_compose(category, merchant, trigger, customer)

    # Validate and clean
    output, warnings = validate_compose_output(output, category, merchant, trigger, customer)
    if warnings:
        logger.info(f"Validation warnings: {warnings}")

    return output


def _fallback_compose(category: dict, merchant: dict, trigger: dict,
                       customer: dict | None = None) -> dict:
    """
    Deterministic fallback when LLM is unavailable.
    Produces a safe, context-anchored message without LLM.
    """
    identity = merchant.get("identity", {})
    name = identity.get("name", "there")
    owner = identity.get("owner_first_name", "")
    kind = trigger.get("kind", "update")
    performance = merchant.get("performance", {})
    views = performance.get("views", "")

    # Determine salutation
    cat_slug = category.get("slug", "")
    if cat_slug == "dentists" and owner:
        salutation = f"Dr. {owner}"
    elif owner:
        salutation = owner
    else:
        salutation = name

    # Build a simple context-anchored message
    if kind == "perf_dip":
        payload = trigger.get("payload", {})
        metric = payload.get("metric", "performance")
        delta = payload.get("delta_pct", -0.3)
        body = f"Hi {salutation}, your {metric} dropped {abs(delta)*100:.0f}% this week. Want me to run a quick diagnostic?"
        cta = "binary_yes_stop"
    elif kind == "renewal_due":
        payload = trigger.get("payload", {})
        days = payload.get("days_remaining", "few")
        body = f"Hi {salutation}, your subscription expires in {days} days. Reply YES to renew."
        cta = "binary_yes_stop"
    elif kind == "research_digest":
        digest = category.get("digest", [])
        if digest:
            item = digest[0]
            body = f"Dr. {owner or name}, {item.get('title', 'JIDA Oct 2026')} ({item.get('source', 'JIDA')}) — {item.get('stat', 'trial results')}. Want me to pull the abstract?"
        else:
            body = f"Hi {salutation}, new clinical digest updates available. Want the summary?"
        cta = "open_ended"
    elif kind == "regulation_change":
        payload = trigger.get("payload", {})
        title = payload.get("title", "DCI Circular")
        deadline = payload.get("deadline_iso", "soon")
        body = f"Dr. {owner or name}, {title} compliance deadline is {deadline}. Want me to check your clinic's setup?"
        cta = "binary_yes_stop"
    elif kind == "recall_due" or customer or trigger.get("scope") == "customer":
        payload = trigger.get("payload", {})
        service = payload.get("service_due", "routine check-up").replace("_", " ")
        slots = payload.get("available_slots", [])
        slot_text = f" Open slots: {slots[0].get('label', '')} or {slots[1].get('label', '')}." if len(slots) >= 2 else ""
        cust_name = customer.get("identity", {}).get("name") if customer else None
        if not cust_name:
            cid = trigger.get("customer_id", "")
            parts = cid.split("_")
            cust_name = parts[2].title() if len(parts) >= 3 else "there"
        body = f"Hi {cust_name}, {name} here. Time for your {service}!{slot_text} Reply YES to book or STOP to pause."
        cta = "binary_yes_stop"
        send_as = "merchant_on_behalf"
    else:
        body = f"Hi {salutation}, you have {views} views this month. Want to see how that compares to peers in your area?"
        cta = "open_ended"

    send_as = "merchant_on_behalf" if (customer or trigger.get("scope") == "customer" or kind == "recall_due") else "vera"

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", ""),
        "rationale": f"Fallback composition for {kind} trigger (LLM unavailable)",
    }
