# Vera AI Challenge — Submission by Amulya Kaushik

## Live Deployment
- **Base URL**: `https://vera-magicpin-bot-chtq.onrender.com`
- **Health Check**: `https://vera-magicpin-bot-chtq.onrender.com/v1/healthz`
- **Metadata**: `https://vera-magicpin-bot-chtq.onrender.com/v1/metadata`
- **Interactive OpenAPI Docs**: `https://vera-magicpin-bot-chtq.onrender.com/docs`

## Approach

**Architecture**: Stateful FastAPI server implementing the 5 required endpoints (`/v1/healthz`, `/v1/metadata`, `/v1/context`, `/v1/tick`, `/v1/reply`) backed by trigger-routed prompt composition, deterministic multi-turn reply handling, and dual-provider LLM failover.

### Core Design

1. **Trigger-Routed Prompting**: Instead of one generic prompt, the composer dispatches to 15+ trigger-kind-specific prompt templates. A `research_digest` trigger gets clinical/citation-focused instructions. A `perf_dip` trigger gets loss-aversion framing. A `recall_due` trigger gets customer-facing slot-offering instructions. This maximizes Trigger Relevance and Specificity scores.

2. **Rubric-Aware System Prompt**: The system prompt explicitly encodes all 5 scoring dimensions (Specificity, Category Fit, Merchant Fit, Trigger Relevance, Engagement Compulsion) with DO/DON'T examples. The LLM knows exactly what the judge looks for.

3. **Deterministic Fast-Paths & Auto-Reply Detection**:
   - Uses a dictionary of 15+ known WhatsApp Business canned phrases plus message-repetition counting.
   - First auto-reply → human-redirect attempt. Second → graceful exit (`end`).
   - Hostile/unsubscribe signals ("stop", "unsubscribe", "spam") → immediate graceful termination (`end`).
   - Commitment transitions ("yes", "let's do it", "proceed") → instant switch to `send` Action mode (avoids the Pattern D loop).

4. **Resilient Dual-Provider LLM Architecture**:
   - **Primary**: Groq (`qwen/qwen3.8-27b`) — Ultra-low latency (~0.6-1.1s) for fast response times.
   - **Resilient Secondary**: Google Gemini (`gemini-3.5-flash-lite`) — Automatically intercepts HTTP 429 rate limits or provider downtime, ensuring zero dropouts during extended stress tests.
   - **Deterministic Fallback**: Context-grounded template fallback ensuring the bot never returns empty even during catastrophic API outages.

5. **Tick Pacing & Action Prioritization (Strategy A)**:
   - Evaluates incoming triggers by urgency tiers (Tier 5: `perf_dip`, `renewal_due`, `dispute`; Tier 4: `lead_received`, `bill_upload`, etc.).
   - Paces tick execution to a maximum of 3 high-impact actions per tick, keeping token burn well within rate limits and response times under 3.5 seconds (far below the evaluator's 15s timeout).
   - Enforces per-merchant throttling (at most 1 message per merchant per tick) to prevent spamming.

6. **Post-Validation Pipeline**: Every LLM output is validated for taboo words, CTA shape, send_as correctness, anti-repetition, and hallucination guards before being returned.

## Evaluation Results

- **Evaluator Suite (`judge_simulator.py all`)**: PASS across all scenarios (`warmup`, `auto_reply`, `intent`, `hostile`).
- **Live Scoring (`judge_simulator.py phase2_short`)**: **45/50 (90%) EXCELLENT**
  - **Category Fit**: 10/10
  - **Specificity**: 9/10
  - **Merchant Fit**: 9/10
  - **Engagement**: 9/10
  - **Decision Quality**: 8/10

## Tradeoffs

| Decision | Tradeoff |
|---|---|
| Single LLM call per compose (no chaining) | Speed over depth — but trigger-routed templates compensate by providing kind-specific instructions |
| In-memory state only | Simple, ultra-fast, and thread-safe. Perfectly suited for the 60-min test window |
| Dual-provider failover (Groq + Gemini) | Instant failover without sleep penalties vs. managing two API keys |
| Tick pacing (cap at top 3 actions) | Paces quota consumption across the 60-min window while prioritizing high-urgency business triggers |
| Hindi-English code-mix via prompt instructions | Generates natural conversational Hinglish tailored to merchant/customer preference |

## What Additional Context Would Have Helped

1. **Real merchant reply distributions** — knowing that 70% of replies are auto-replies helped tune detection thresholds, and real data on edge-case phrasing would enhance it further.
2. **Actual WhatsApp template approval constraints** — template structures in production constrain the first-touch message format.
3. **Per-category engagement rate baselines** — to calibrate how aggressive vs. restrained the CTA should be across different business verticals.

## File Structure

- `bot.py` — FastAPI server (5 core endpoints + root + teardown)
- `composer.py` — Core `compose()` 4-context composition engine
- `reply_handler.py` — Multi-turn response logic & intent classification
- `state.py` — In-memory thread-safe state management with atomic versioning
- `llm_client.py` — Dual-provider LLM client (Groq primary + Gemini fallback)
- `validators.py` — Post-LLM validation and anti-hallucination sanitization
- `prompts/system.py` — Rubric-aligned system prompts
- `prompts/trigger_templates.py` — Per-trigger-kind templates across 15+ trigger kinds
- `generate_submission.py` — Generates canonical `submission.jsonl`
- `submission.jsonl` — 30 canonical test pair outputs

