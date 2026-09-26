#!/usr/bin/env python3
"""
Generate submission.jsonl from the 30 canonical test pairs.

Usage:
    export LLM_PROVIDER=groq
    export LLM_API_KEY=your_key_here
    python generate_submission.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from composer import compose

DATASET_DIR = Path(__file__).parent / "dataset"
EXPANDED_DIR = DATASET_DIR / "expanded"


def load_expanded_dataset():
    """Load the expanded dataset."""
    categories = {}
    for f in (EXPANDED_DIR / "categories").glob("*.json"):
        data = json.load(open(f))
        categories[data.get("slug", f.stem)] = data

    merchants = {}
    for f in (EXPANDED_DIR / "merchants").glob("*.json"):
        data = json.load(open(f))
        merchants[data.get("merchant_id", f.stem)] = data

    customers = {}
    for f in (EXPANDED_DIR / "customers").glob("*.json"):
        data = json.load(open(f))
        customers[data.get("customer_id", f.stem)] = data

    triggers = {}
    for f in (EXPANDED_DIR / "triggers").glob("*.json"):
        data = json.load(open(f))
        triggers[data.get("id", f.stem)] = data

    return categories, merchants, customers, triggers


def main():
    # Load dataset
    print("Loading expanded dataset...")
    categories, merchants, customers, triggers = load_expanded_dataset()
    print(f"  Loaded: {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers")

    # Load test pairs
    test_pairs_path = EXPANDED_DIR / "test_pairs.json"
    if not test_pairs_path.exists():
        print(f"ERROR: {test_pairs_path} not found. Run generate_dataset.py first.")
        sys.exit(1)

    with open(test_pairs_path) as f:
        test_pairs = json.load(f)["pairs"]

    print(f"  {len(test_pairs)} test pairs to process")

    # Process each test pair
    results = []
    for i, pair in enumerate(test_pairs):
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        print(f"\n[{i+1}/{len(test_pairs)}] {test_id}: {trigger_id}")

        # Resolve contexts
        trigger = triggers.get(trigger_id)
        merchant = merchants.get(merchant_id)
        if not trigger or not merchant:
            print(f"  SKIP — trigger or merchant not found")
            continue

        category_slug = merchant.get("category_slug", "")
        category = categories.get(category_slug)
        if not category:
            print(f"  SKIP — category {category_slug} not found")
            continue

        customer = customers.get(customer_id) if customer_id else None

        # Compose
        try:
            result = compose(category, merchant, trigger, customer)
            print(f"  OK — {result.get('body', '')[:60]}...")
        except Exception as e:
            print(f"  ERROR — {e}")
            result = {
                "body": f"Hi {merchant.get('identity', {}).get('name', 'there')}, update available for your business.",
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": f"Fallback — compose error: {str(e)[:50]}",
            }

        # Build submission line
        line = {
            "test_id": test_id,
            "body": result.get("body", ""),
            "cta": result.get("cta", "open_ended"),
            "send_as": result.get("send_as", "vera"),
            "suppression_key": result.get("suppression_key", ""),
            "rationale": result.get("rationale", ""),
        }
        results.append(line)

    # Write submission.jsonl
    output_path = Path(__file__).parent / "submission.jsonl"
    with open(output_path, "w") as f:
        for line in results:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Done! Wrote {len(results)} lines to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
