"""
Per-trigger-kind prompt templates.

Each template focuses the LLM on extracting the right data from context
and framing the message to maximize the scoring dimension most relevant
to that trigger type.
"""

from __future__ import annotations
import json
from typing import Optional


def build_compose_prompt(category: dict, merchant: dict, trigger: dict,
                         customer: dict | None = None) -> str:
    """
    Build a structured user prompt for the compose() call.
    Dispatches to trigger-kind-specific instructions.
    """
    kind = trigger.get("kind", "unknown")

    # Base context block — always included
    context_block = _build_context_block(category, merchant, trigger, customer)

    # Kind-specific instructions
    kind_instructions = _get_kind_instructions(kind, trigger, category, merchant, customer)

    return f"""{context_block}

## TRIGGER-SPECIFIC INSTRUCTIONS
{kind_instructions}

Compose the message now. Return ONLY the JSON object."""


def _build_context_block(category: dict, merchant: dict, trigger: dict,
                          customer: dict | None) -> str:
    """Assemble the structured context block the LLM needs."""

    # Category context
    voice = category.get("voice", {})
    peer_stats = category.get("peer_stats", {})
    offer_catalog = category.get("offer_catalog", [])
    digest = category.get("digest", [])
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])

    # Merchant context
    identity = merchant.get("identity", {})
    performance = merchant.get("performance", {})
    offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    customer_agg = merchant.get("customer_aggregate", {})
    review_themes = merchant.get("review_themes", [])
    conv_history = merchant.get("conversation_history", [])[-3:]  # Last 3 turns
    subscription = merchant.get("subscription", {})

    # Trigger context
    trigger_payload = trigger.get("payload", {})

    # Match digest item if trigger references one
    top_item_id = trigger_payload.get("top_item_id")
    matched_digest = None
    if top_item_id:
        for d in digest:
            if d.get("id") == top_item_id:
                matched_digest = d
                break

    parts = [
        "## CONTEXT FOR COMPOSITION",
        "",
        f"### Category: {category.get('slug', 'unknown')}",
        f"- Voice tone: {voice.get('tone', 'professional')}",
        f"- Vocab taboo (NEVER use these): {voice.get('vocab_taboo', [])}",
        f"- Salutation style: {voice.get('salutation_examples', [])}",
        f"- Offer catalog: {json.dumps([o.get('title') for o in offer_catalog[:5]])}",
        f"- Peer stats: avg_rating={peer_stats.get('avg_rating', '?')}, avg_reviews={peer_stats.get('avg_review_count', peer_stats.get('avg_reviews', '?'))}, avg_ctr={peer_stats.get('avg_ctr', '?')}, avg_views_30d={peer_stats.get('avg_views_30d', '?')}",
    ]

    if matched_digest:
        parts.append(f"- RELEVANT DIGEST ITEM: {json.dumps(matched_digest)}")
    elif digest:
        parts.append(f"- Digest items available: {json.dumps([d.get('title') for d in digest[:3]])}")

    if seasonal:
        parts.append(f"- Seasonal beats: {json.dumps(seasonal[:3])}")
    if trends:
        parts.append(f"- Trend signals: {json.dumps(trends[:3])}")

    parts.extend([
        "",
        f"### Merchant: {identity.get('name', 'Unknown')}",
        f"- Owner: {identity.get('owner_first_name', 'Unknown')}",
        f"- Location: {identity.get('locality', '?')}, {identity.get('city', '?')}",
        f"- Languages: {identity.get('languages', ['en'])}",
        f"- Verified: {identity.get('verified', False)}",
        f"- Subscription: {subscription.get('status', '?')} ({subscription.get('plan', '?')}), {subscription.get('days_remaining', '?')} days remaining",
        f"- Performance (30d): views={performance.get('views', '?')}, calls={performance.get('calls', '?')}, directions={performance.get('directions', '?')}, ctr={performance.get('ctr', '?')}",
        f"- 7d delta: views {_fmt_pct(performance.get('delta_7d', {}).get('views_pct'))}, calls {_fmt_pct(performance.get('delta_7d', {}).get('calls_pct'))}",
        f"- Active offers: {json.dumps([o.get('title') for o in offers]) if offers else 'None'}",
        f"- Signals: {signals}",
        f"- Customer aggregate: {json.dumps(customer_agg)}",
    ])

    if review_themes:
        parts.append(f"- Review themes: {json.dumps(review_themes[:3])}")

    if conv_history:
        parts.append(f"- Recent conversation (last {len(conv_history)} turns):")
        for turn in conv_history:
            parts.append(f"  [{turn.get('from', '?')}] {turn.get('body', '')[:100]}")

    parts.extend([
        "",
        f"### Trigger",
        f"- Kind: {trigger.get('kind', 'unknown')}",
        f"- Source: {trigger.get('source', 'unknown')} ({trigger.get('scope', 'merchant')}-scoped)",
        f"- Urgency: {trigger.get('urgency', '?')}/5",
        f"- Payload: {json.dumps(trigger_payload)}",
        f"- Suppression key: {trigger.get('suppression_key', '')}",
    ])

    if customer:
        cust_identity = customer.get("identity", {})
        cust_rel = customer.get("relationship", {})
        parts.extend([
            "",
            f"### Customer: {cust_identity.get('name', 'Unknown')}",
            f"- Language pref: {cust_identity.get('language_pref', 'en')}",
            f"- State: {customer.get('state', 'unknown')}",
            f"- Visits: {cust_rel.get('visits_total', '?')} total, last on {cust_rel.get('last_visit', '?')}",
            f"- Services received: {cust_rel.get('services_received', [])}",
            f"- Preferences: {json.dumps(customer.get('preferences', {}))}",
            f"- Consent scope: {customer.get('consent', {}).get('scope', [])}",
        ])

    return "\n".join(parts)


def _fmt_pct(val) -> str:
    if val is None:
        return "?"
    sign = "+" if val > 0 else ""
    return f"{sign}{val*100:.0f}%"


# ─── Per-kind instructions ───────────────────────────────────────────────────

def _get_kind_instructions(kind: str, trigger: dict, category: dict,
                            merchant: dict, customer: dict | None) -> str:
    """Return trigger-kind-specific composition instructions."""
    dispatch = {
        "research_digest": _inst_research_digest,
        "regulation_change": _inst_regulation_change,
        "perf_dip": _inst_perf_dip,
        "perf_spike": _inst_perf_spike,
        "recall_due": _inst_recall_due,
        "festival_upcoming": _inst_festival,
        "renewal_due": _inst_renewal,
        "dormant_with_vera": _inst_dormant,
        "curious_ask_due": _inst_curious_ask,
        "review_theme_emerged": _inst_review_theme,
        "competitor_opened": _inst_competitor,
        "milestone_reached": _inst_milestone,
        "customer_lapsed_soft": _inst_customer_lapsed,
        "customer_lapsed_hard": _inst_customer_lapsed,
        "appointment_tomorrow": _inst_appointment,
        "wedding_package_followup": _inst_wedding_followup,
        "chronic_refill_due": _inst_refill,
        "trial_followup": _inst_trial_followup,
    }

    fn = dispatch.get(kind, _inst_default)
    return fn(trigger, category, merchant, customer)


def _inst_research_digest(trigger, category, merchant, customer) -> str:
    return """This is a RESEARCH DIGEST trigger. Frame the message as a peer sharing relevant clinical/industry research.

KEY SCORING TARGETS:
- SPECIFICITY: Cite the exact study — trial size, percentage, source publication, page number from the digest. CRITICAL: Do NOT invent patient counts or chart numbers (e.g. do not invent "124 patients in your chart") unless an exact number appears in the merchant context.
- CATEGORY FIT: Use clinical/peer vocabulary, source citation style
- TRIGGER RELEVANCE: The digest item IS the reason for messaging — make that explicit
- ENGAGEMENT: Offer to pull the abstract or draft patient-ed content they can reshare

DO: "Dr. Meera, JIDA Oct 2026 (p.14) trial (n=2,100) shows 38% lower caries recurrence with 3-month fluoride varnish recalls. Want me to pull the 1-page summary?"
DON'T: "New research shows dental care is important"
CTA should be open_ended."""


def _inst_regulation_change(trigger, category, merchant, customer) -> str:
    payload = trigger.get("payload", {})
    deadline = payload.get("deadline_iso", "upcoming")
    return f"""This is a REGULATION CHANGE trigger. Frame with urgency — there's a compliance deadline ({deadline}).

KEY SCORING TARGETS:
- SPECIFICITY: Name the exact regulation, regulator, effective date
- TRIGGER RELEVANCE: The regulatory change IS the urgency — don't bury it
- ENGAGEMENT: Offer to summarize requirements or check their current compliance

DO: "DCI revised radiograph dose limits — effective Dec 15, 2026. Want me to check if your setup needs updates?"
DON'T: "Regulations are changing, stay updated"
CTA should be binary_yes_stop for action-oriented compliance check."""


def _inst_perf_dip(trigger, category, merchant, customer) -> str:
    payload = trigger.get("payload", {})
    metric = payload.get("metric", "calls")
    delta = payload.get("delta_pct", -0.5)
    baseline = payload.get("vs_baseline", "?")
    return f"""This is a PERFORMANCE DIP trigger. The merchant's {metric} dropped {abs(delta)*100:.0f}% vs baseline of {baseline}.

KEY SCORING TARGETS:
- SPECIFICITY: State the exact metric, exact percentage drop, exact baseline number
- MERCHANT FIT: Use their actual performance numbers
- ENGAGEMENT: Loss aversion framing — "you're losing X". Offer a specific fix.

DO: "Calls dropped 50% this week (4 vs your usual 12). 3 things I can do right now to reverse this."
DON'T: "Your performance could be better"
CTA should be binary_yes_stop — "Want me to run a quick diagnostic?"."""


def _inst_perf_spike(trigger, category, merchant, customer) -> str:
    return """This is a PERFORMANCE SPIKE trigger. Celebrate momentum and suggest how to amplify it.

KEY SCORING TARGETS:
- SPECIFICITY: State the exact metric improvement with numbers
- ENGAGEMENT: Positive reinforcement + curiosity about what's driving it

DO: "Views up 28% this week — something's clicking. Want to see which keywords are driving the spike?"
DON'T: "Great job, keep it up!"
CTA should be open_ended."""


def _inst_recall_due(trigger, category, merchant, customer) -> str:
    payload = trigger.get("payload", {})
    slots = payload.get("available_slots", [])
    slot_text = " or ".join([s.get("label", "?") for s in slots[:2]]) if slots else "this week"
    return f"""This is a RECALL DUE trigger — customer-facing message sent on behalf of the merchant.

CRITICAL: Set send_as to "merchant_on_behalf" — this goes to the patient/customer from the merchant's number.

KEY SCORING TARGETS:
- SPECIFICITY: Name the service due, time since last visit, specific slot options ({slot_text})
- CUSTOMER FIT: Use customer's name, language preference, match their preferred slot times
- MERCHANT FIT: Use the merchant's actual offer prices from their active offers

DO: "Hi Priya, Dr. Meera's clinic here 🦷 6-month cleaning recall due. 2 slots: Wed 5pm or Thu 6pm. ₹299 cleaning. Reply 1 or 2."
DON'T: "It's time for your dental checkup"
CTA: For booking flows, numbered slot selection is acceptable."""


def _inst_festival(trigger, category, merchant, customer) -> str:
    payload = trigger.get("payload", {})
    festival = payload.get("festival", "upcoming festival")
    days = payload.get("days_until", "?")
    return f"""This is a FESTIVAL trigger — {festival} is {days} days away.

KEY SCORING TARGETS:
- TRIGGER RELEVANCE: Name the festival and days until it
- CATEGORY FIT: Make the festival connection category-appropriate
- ENGAGEMENT: Offer to draft a campaign/post — effort externalization

DO: "Diwali in 4 days — want me to draft a festive Google post + a WhatsApp campaign for your regulars?"
DON'T: "Happy upcoming festival!"
CTA should be binary_yes_stop."""


def _inst_renewal(trigger, category, merchant, customer) -> str:
    payload = trigger.get("payload", {})
    days = payload.get("days_remaining", "?")
    plan = payload.get("plan", "Pro")
    amount = payload.get("renewal_amount", "?")
    return f"""This is a RENEWAL trigger — subscription expires in {days} days.

KEY SCORING TARGETS:
- SPECIFICITY: Days remaining ({days}), plan name ({plan}), renewal amount (₹{amount})
- MERCHANT FIT: Reference what they'd lose — their performance numbers, active offers
- ENGAGEMENT: Loss aversion — "your listing goes dark" / "active offers expire"

DO: "Pro expires in 12 days — your 2,410 monthly views and Dental Cleaning offer go dark. Renew at ₹4,999? Reply YES."
DON'T: "Your subscription is expiring soon"
CTA should be binary_yes_stop."""


def _inst_dormant(trigger, category, merchant, customer) -> str:
    return """This is a DORMANT trigger — merchant hasn't engaged with Vera in 14+ days.

KEY SCORING TARGETS:
- ENGAGEMENT: Re-engagement through curiosity or a value-first observation
- MERCHANT FIT: Share something specific about their account they'd want to know
- SPECIFICITY: Lead with a concrete data point about their performance

DO: "Quick update — your views hit 2,410 this month (above average for Lajpat Nagar). Also noticed 3 reviews mention wait times. Want the details?"
DON'T: "We haven't heard from you in a while!"
CTA should be open_ended."""


def _inst_curious_ask(trigger, category, merchant, customer) -> str:
    return """This is a CURIOUS ASK trigger — ask the merchant a genuine question about their business.

KEY SCORING TARGETS:
- ENGAGEMENT: Asking the merchant is a powerful compulsion lever. Make it a question they'd enjoy answering.
- CATEGORY FIT: The question should be category-appropriate and practical
- MERCHANT FIT: Reference something specific about their business

DO: "Quick question — what's your most-requested service this week? Helps me suggest the right offer to feature."
DON'T: "How is your business doing?"
CTA should be open_ended."""


def _inst_review_theme(trigger, category, merchant, customer) -> str:
    return """This is a REVIEW THEME trigger — a pattern emerged from recent customer reviews.

KEY SCORING TARGETS:
- SPECIFICITY: Name the theme, number of occurrences, and quote a real review snippet
- MERCHANT FIT: Show you've read their reviews
- ENGAGEMENT: Frame as actionable intelligence, not criticism

DO: "3 reviews this week mention 'wait time' — one said 'had to wait 30 min on Sunday afternoon'. Want me to draft a Sunday-slot Google post to set expectations?"
DON'T: "Your reviews could use some attention"
CTA should be open_ended."""


def _inst_competitor(trigger, category, merchant, customer) -> str:
    return """This is a COMPETITOR OPENED trigger — a new competitor appeared nearby.

KEY SCORING TARGETS:
- SPECIFICITY: Distance, type of competitor
- ENGAGEMENT: Voyeur curiosity — "want to see their profile?" / "want to see how you compare?"
- MERCHANT FIT: Position their strengths vs the competitive threat

DO: "New dental clinic opened 1.3km from you on GBP. Your 4.4★ rating (298 reviews) is strong. Want to see their profile?"
DON'T: "Competition is increasing in your area"
CTA should be open_ended."""


def _inst_milestone(trigger, category, merchant, customer) -> str:
    return """This is a MILESTONE trigger — the merchant hit a significant achievement.

KEY SCORING TARGETS:
- SPECIFICITY: Name the exact milestone with numbers
- ENGAGEMENT: Celebrate + suggest how to leverage it (social proof, post, share)

DO: "Crossed 100 reviews! 🎉 You're in the top 15% of salons in HSR Layout. Want me to draft a celebratory Google post?"
DON'T: "Congratulations on your achievement!"
CTA should be binary_yes_stop."""


def _inst_customer_lapsed(trigger, category, merchant, customer) -> str:
    return """This is a CUSTOMER LAPSED trigger — customer-facing win-back message.

CRITICAL: Set send_as to "merchant_on_behalf".

KEY SCORING TARGETS:
- CUSTOMER FIT: Use their name, last service, time since visit
- MERCHANT FIT: Use actual offers and prices
- ENGAGEMENT: Low-friction re-booking, mention what they've used before

DO: "Hi Priya, it's been 5 months since your last cleaning at Dr. Meera's. Due for a check? ₹299 cleaning this week. Reply YES to book."
DON'T: "We miss you! Come back for a visit."
CTA should be binary_yes_stop."""


def _inst_appointment(trigger, category, merchant, customer) -> str:
    return """This is an APPOINTMENT TOMORROW trigger — customer-facing reminder.

CRITICAL: Set send_as to "merchant_on_behalf".

KEY SCORING TARGETS:
- SPECIFICITY: Date, time, service, location
- CUSTOMER FIT: Name, preferences

DO: "Hi Priya, reminder: cleaning appointment tomorrow (Wed 6pm) at Dr. Meera's, Lajpat Nagar. ₹299. See you there!"
DON'T: "Don't forget your appointment"
CTA should be none."""


def _inst_wedding_followup(trigger, category, merchant, customer) -> str:
    payload = trigger.get("payload", {})
    wedding_date = payload.get("wedding_date", "upcoming")
    days = payload.get("days_to_wedding", "?")
    next_step = payload.get("next_step_window_open", "next step")
    return f"""This is a WEDDING PACKAGE FOLLOWUP trigger — customer has a wedding on {wedding_date} ({days} days away).

CRITICAL: Set send_as to "merchant_on_behalf".

KEY SCORING TARGETS:
- CUSTOMER FIT: Reference their trial, wedding date, next step ({next_step})
- SPECIFICITY: Exact dates, service names
- ENGAGEMENT: Timeline urgency for bridal prep

DO: "Hi Kavya! Wedding in 196 days 🎊 Your bridal trial was lovely. Time to start the 30-day skin prep program. Want to book the first session?"
DON'T: "Congratulations on your upcoming wedding"
CTA should be open_ended."""


def _inst_refill(trigger, category, merchant, customer) -> str:
    return """This is a CHRONIC REFILL DUE trigger — customer-facing medication/service refill reminder.

CRITICAL: Set send_as to "merchant_on_behalf".

KEY SCORING TARGETS:
- SPECIFICITY: Name the medication/service, last refill date, expected due date
- CUSTOMER FIT: Use their name, preferences
- ENGAGEMENT: Convenience — "we have it ready" / "order now, pick up today"

CTA should be binary_yes_stop."""


def _inst_trial_followup(trigger, category, merchant, customer) -> str:
    return """This is a TRIAL FOLLOWUP trigger — follow up on a customer's trial/first visit.

CRITICAL: Set send_as to "merchant_on_behalf".

KEY SCORING TARGETS:
- CUSTOMER FIT: Reference their trial experience, service they tried
- ENGAGEMENT: Ask about their experience, offer next step

CTA should be open_ended."""


def _inst_default(trigger, category, merchant, customer) -> str:
    kind = trigger.get("kind", "unknown")
    return f"""This is a {kind} trigger. No specialized template available.

GENERAL APPROACH:
- Lead with the most specific, verifiable fact from the trigger payload
- Match the category voice
- Personalize to the merchant's current state
- End with a clear, low-friction CTA

CTA should be open_ended unless there's a clear binary action to take."""


# ─── Reply prompt builder ─────────────────────────────────────────────────────

def build_reply_prompt(conversation: list[dict], merchant_message: str,
                       merchant: dict | None, category: dict | None,
                       trigger_context: str = "") -> str:
    """Build the user prompt for the reply handler."""

    conv_text = ""
    for turn in conversation[-6:]:  # Last 6 turns max
        role = turn.get("from", "?")
        body = turn.get("body", "")
        conv_text += f"[{role}] {body}\n"

    merchant_name = "the merchant"
    merchant_info = ""
    if merchant:
        identity = merchant.get("identity", {})
        merchant_name = identity.get("name", "the merchant")
        merchant_info = f"""
Merchant: {merchant_name}
Owner: {identity.get('owner_first_name', '?')}
Category: {merchant.get('category_slug', '?')}
Languages: {identity.get('languages', ['en'])}
Signals: {merchant.get('signals', [])}
"""

    return f"""## CONVERSATION HISTORY
{conv_text}
## LATEST MERCHANT MESSAGE
[merchant] {merchant_message}

## MERCHANT INFO
{merchant_info}

{trigger_context}

Determine your next action. Return ONLY the JSON object."""
