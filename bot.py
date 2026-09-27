"""
magicpin Vera AI Challenge — Bot Server

FastAPI server implementing the 5 required endpoints:
  - GET  /v1/healthz
  - GET  /v1/metadata
  - POST /v1/context
  - POST /v1/tick
  - POST /v1/reply

Run: uvicorn bot:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import state
from composer import compose
from reply_handler import handle_reply

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bot")

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vera AI Challenge Bot",
    description="magicpin Vera merchant assistant — challenge submission",
    version="1.0.0",
)

# ─── Payload size middleware ──────────────────────────────────────────────────

MAX_PAYLOAD_BYTES = 500 * 1024  # 500 KB


@app.middleware("http")
async def check_payload_size(request: Request, call_next):
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_PAYLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": "Payload too large", "max_bytes": MAX_PAYLOAD_BYTES},
            )
    return await call_next(request)


# ═════════════════════════════════════════════════════════════════════════════
# ROOT: GET /
# ═════════════════════════════════════════════════════════════════════════════


@app.get("/")
async def root():
    return {
        "service": "magicpin Vera AI Merchant Assistant",
        "team": "Amulya Kaushik",
        "status": "online",
        "endpoints": {
            "healthz": "/v1/healthz",
            "metadata": "/v1/metadata",
            "context": "/v1/context",
            "tick": "/v1/tick",
            "reply": "/v1/reply",
            "docs": "/docs",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1: GET /v1/healthz
# ═════════════════════════════════════════════════════════════════════════════


@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": state.get_uptime(),
        "contexts_loaded": state.get_context_counts(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2: GET /v1/metadata
# ═════════════════════════════════════════════════════════════════════════════


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Amulya Kaushik",
        "team_members": ["Amulya Kaushik"],
        "model": "qwen/qwen3.8-27b (Groq) with Gemini 3.5 Flash Lite Failover",
        "approach": "Trigger-routed single-prompt composer with rubric-aware system prompt, deterministic auto-reply detection, and LLM-powered intent classification",
        "contact_email": "amulyakaushik7@gmail.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }



# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3: POST /v1/context
# ═════════════════════════════════════════════════════════════════════════════


class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: ContextRequest):
    result = state.push_context(body.scope, body.context_id, body.version, body.payload)

    if result.get("accepted") is False and result.get("reason") == "stale_version":
        raise HTTPException(status_code=409, detail=result)
    elif result.get("accepted") is False:
        raise HTTPException(status_code=400, detail=result)

    logger.info(f"Context stored: {body.scope}/{body.context_id} v{body.version}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4: POST /v1/tick
# ═════════════════════════════════════════════════════════════════════════════

# Trigger urgency tiers for tick pacing (Strategy A)
_TRIGGER_KIND_PRIORITY = {
    # Critical business urgency (Tier 5)
    "perf_dip": 5,
    "renewal_due": 5,
    "dispute": 5,
    "payment_failed": 5,
    "low_rating": 5,
    # High responsiveness urgency (Tier 4)
    "lead_received": 4,
    "bill_upload": 4,
    "campaign_expiring": 4,
    # Medium engagement urgency (Tier 3)
    "merchant_inactivity": 3,
    "catalogue_view_spike": 3,
    "milestone_reached": 3,
}


def _get_trigger_priority(trg_id: str) -> tuple[int, str]:
    trg = state.get_trigger(trg_id)
    if not trg:
        return (-1, trg_id)
    urgency = trg.get("urgency")
    if urgency is not None and isinstance(urgency, (int, float)):
        return (int(urgency), trg_id)
    kind = trg.get("kind", "")
    return (_TRIGGER_KIND_PRIORITY.get(kind, 2), trg_id)


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickRequest):
    actions = []
    start_time = time.time()
    contacted_merchants = set()

    # Strategy A: Sort triggers by urgency so highest-impact triggers are handled first
    sorted_triggers = sorted(body.available_triggers, key=_get_trigger_priority, reverse=True)

    for trg_id in sorted_triggers:
        # Budget check — judge simulator has a strict 15s timeout
        elapsed = time.time() - start_time
        if elapsed > 10.0:
            logger.warning(f"Tick time budget reached at {elapsed:.1f}s, returning {len(actions)} actions")
            break

        # Strategy A: Cap at 3 actions per tick to pace LLM rate limits across the 60-min window
        if len(actions) >= 3:
            break

        # Look up trigger
        trigger = state.get_trigger(trg_id)
        if not trigger:
            logger.debug(f"Trigger {trg_id} not found in store, skipping")
            continue

        # Check suppression
        suppression_key = trigger.get("suppression_key", "")
        if suppression_key and state.is_suppressed(suppression_key):
            logger.debug(f"Trigger {trg_id} suppressed by key {suppression_key}")
            continue

        # Resolve merchant
        merchant_id = trigger.get("merchant_id")
        if not merchant_id:
            continue

        # Avoid spamming the same merchant multiple times within a single tick
        if merchant_id in contacted_merchants:
            logger.debug(f"Merchant {merchant_id} already contacted in this tick, deferring trigger {trg_id}")
            continue

        merchant = state.get_merchant(merchant_id)
        if not merchant:
            logger.debug(f"Merchant {merchant_id} not found for trigger {trg_id}")
            continue

        # Resolve category
        category_slug = merchant.get("category_slug", "")
        category = state.get_category(category_slug)
        if not category:
            logger.debug(f"Category {category_slug} not found for merchant {merchant_id}")
            continue

        # Resolve optional customer
        customer_id = trigger.get("customer_id")
        customer = state.get_customer(customer_id) if customer_id else None

        # ─── Compose the message ──────────────────────────────────────────
        try:
            result = compose(category, merchant, trigger, customer)
        except Exception as e:
            logger.error(f"Compose failed for {trg_id}: {e}")
            continue

        body_text = result.get("body", "")
        if not body_text or not body_text.strip():
            logger.warning(f"Empty body from compose for trigger {trg_id}")
            continue

        # Build the action
        conversation_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:8]}"

        action = {
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [
                merchant.get("identity", {}).get("name", ""),
                trigger.get("kind", ""),
                body_text[:50],
            ],
            "body": body_text,
            "cta": result.get("cta", "open_ended"),
            "suppression_key": result.get("suppression_key", suppression_key),
            "rationale": result.get("rationale", ""),
        }

        actions.append(action)
        contacted_merchants.add(merchant_id)

        # Mark suppressed
        if suppression_key:
            state.mark_suppressed(suppression_key)

        # Track in conversation history
        state.append_conversation(conversation_id, "vera", body_text)

        logger.info(f"Action composed: {trg_id} → {merchant_id} ({trigger.get('kind', '?')})")

    elapsed = time.time() - start_time
    logger.info(f"Tick completed: {len(actions)} actions in {elapsed:.1f}s")

    return {"actions": actions}


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 5: POST /v1/reply
# ═════════════════════════════════════════════════════════════════════════════


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyRequest):
    logger.info(f"Reply received: conv={body.conversation_id}, turn={body.turn_number}, from={body.from_role}")

    result = handle_reply(
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id or "",
        customer_id=body.customer_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number,
    )

    logger.info(f"Reply action: {result.get('action', '?')} for conv={body.conversation_id}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Optional: POST /v1/teardown (end-of-test cleanup)
# ═════════════════════════════════════════════════════════════════════════════


@app.post("/v1/teardown")
async def teardown():
    """Wipe all in-memory state at end of test."""
    state.clear_all()
    logger.info("Teardown complete — all state wiped")
    return {"status": "torn_down"}
