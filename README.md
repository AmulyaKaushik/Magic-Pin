# Vera AI Challenge — Submission by Amulya Kaushik

## Approach

**Architecture**: Trigger-routed single-prompt composer with rubric-aware system prompt.

The bot is a stateful FastAPI server implementing 5 endpoints (`/v1/healthz`, `/v1/metadata`, `/v1/context`, `/v1/tick`, `/v1/reply`). All context is stored in-memory with version-tracked idempotent updates.

### Core Design

1. **Trigger-Routed Prompting**: Instead of one generic prompt, the composer dispatches to 15+ trigger-kind-specific prompt templates. A `research_digest` trigger gets clinical/citation-focused instructions. A `perf_dip` trigger gets loss-aversion framing. A `recall_due` trigger gets customer-facing slot-offering instructions. This maximizes Trigger Relevance scores.

2. **Rubric-Aware System Prompt**: The system prompt explicitly encodes all 5 scoring dimensions (Specificity, Category Fit, Merchant Fit, Trigger Relevance, Engagement Compulsion) with DO/DON'T examples. The LLM knows exactly what the judge looks for.

3. **Deterministic Auto-Reply Detection**: Uses a dictionary of 15+ known WhatsApp Business canned phrases plus message-repetition counting. No LLM needed — instant detection. First auto-reply → human-redirect attempt. Second → graceful exit.

4. **Intent Classification**: Real merchant replies go through LLM-powered intent detection with structured JSON output. Commitment signals ("yes", "let's do it") immediately switch to action mode — avoiding the Pattern D anti-pattern.

5. **Post-Validation Pipeline**: Every LLM output is validated for taboo words, CTA shape, send_as correctness, anti-repetition, and hallucination guards before being returned.

### Model Choice

**Groq (llama-3.1-70b-versatile)**: Chosen for speed (~1-3s latency) which keeps comfortably within the 30s timeout budget, even allowing a retry on JSON parse failure. Temperature=0 for deterministic output.

## Tradeoffs

| Decision | Tradeoff |
|---|---|
| Single LLM call per compose (no chaining) | Speed over depth — but the trigger-routed templates compensate by providing kind-specific instructions |
| In-memory state only | Simple and fast, but can't survive a process restart. Acceptable for the 60-min test window |
| Post-validation instead of constrained decoding | More flexible, handles edge cases, but adds ~10ms overhead |
| Hindi-English code-mix delegated to LLM | Works well with the system prompt instruction; manual template approach would be more brittle |
| Fallback composer (no-LLM) | Ensures the bot never returns empty — even if the API is down, it produces a context-anchored message |

## What Additional Context Would Have Helped

1. **Real merchant reply distributions** — knowing that 70% of replies are auto-replies would let me tune detection thresholds more precisely
2. **Actual WhatsApp template approval constraints** — the template_name field is cosmetic in this challenge, but in production the template structure constrains the first-touch message format
3. **Per-category engagement rate baselines** — would help calibrate how aggressive vs. restrained the CTA should be (dentists may tolerate more clinical depth; restaurants may need shorter messages)

## File Structure

- `bot.py` — FastAPI server (5 endpoints)
- `composer.py` — Core `compose()` function
- `reply_handler.py` — Multi-turn response logic
- `state.py` — In-memory state management
- `llm_client.py` — LLM API wrapper (Groq)
- `validators.py` — Post-LLM validation
- `prompts/system.py` — System prompts
- `prompts/trigger_templates.py` — Per-trigger-kind templates
- `generate_submission.py` — Generates submission.jsonl
- `submission.jsonl` — 30 test pair outputs
