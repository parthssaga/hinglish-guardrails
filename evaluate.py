"""
Evaluation harness.

Runs the input-side guardrails over the labelled set in
data/test_prompts.json and reports precision, recall, F1, accuracy, and
false-positive rate, broken down by language (English vs Hinglish) and by
attack category. This directly produces the pipeline-level numbers the
research paper needs.

Important: this evaluates the GUARDRAILS, not the LLM, so it runs without
a running LLM and without network access. It only needs the guardrail
models (or their fallbacks).

Usage:
    python evaluate.py
    python evaluate.py --data data/test_prompts.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from config import PipelineConfig
from src.preprocessing.transliterate import Transliterator
from src.guardrails.toxicity import ToxicityGuardrail
from src.guardrails.pii import PIIGuardrail
from src.guardrails.injection import InjectionGuardrail
from src.guardrails.jailbreak import JailbreakGuardrail


def build_input_stack(config):
    translit = Transliterator(config)
    return {
        "toxicity": ToxicityGuardrail(config),
        "pii": PIIGuardrail(config),
        "injection": InjectionGuardrail(config),
        "jailbreak": JailbreakGuardrail(config, transliterator=translit),
    }


def predict_unsafe(stack, text: str, language: str | None = None):
    """Return (is_unsafe, firing_guardrail_or_None)."""
    # PII redacts; treat a PII hit as 'unsafe' for evaluation purposes.
    for name in ("pii", "toxicity", "injection", "jailbreak"):
        # Pass the language to injection so DeBERTa is skipped for Hindi.
        if name == "injection":
            r = stack[name].check(text, language=language)
        else:
            r = stack[name].check(text)
        if r.triggered:
            return True, name
    return False, None


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test_prompts.json")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)
    items = dataset["items"]

    config = PipelineConfig()
    stack = build_input_stack(config)

    # global confusion counts (unsafe == positive class)
    tp = fp = tn = fn = 0
    # breakdowns
    by_lang = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion_rows = []

    print(f"Evaluating {len(items)} prompts...\n")

    for item in items:
        text = item["text"]
        lang = item["language"]
        gold_unsafe = item["expected"] == "unsafe"
        cat = item["category"]

        pred_unsafe, fired = predict_unsafe(stack, text, language=lang)

        # global
        if gold_unsafe and pred_unsafe:
            tp += 1; bucket = "TP"
        elif gold_unsafe and not pred_unsafe:
            fn += 1; bucket = "FN"
        elif not gold_unsafe and pred_unsafe:
            fp += 1; bucket = "FP"
        else:
            tn += 1; bucket = "TN"

        # per language
        L = by_lang[lang]
        if gold_unsafe and pred_unsafe: L["tp"] += 1
        elif gold_unsafe and not pred_unsafe: L["fn"] += 1
        elif not gold_unsafe and pred_unsafe: L["fp"] += 1
        else: L["tn"] += 1

        # per category accuracy
        by_cat[cat]["total"] += 1
        if gold_unsafe == pred_unsafe:
            by_cat[cat]["correct"] += 1

        confusion_rows.append((bucket, lang, cat, fired or "-", text[:50]))

    # ---- report ----------------------------------------------------------
    precision, recall, f1 = prf(tp, fp, fn)
    accuracy = (tp + tn) / len(items) if items else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print("=" * 60)
    print("OVERALL (positive class = unsafe)")
    print("=" * 60)
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  Accuracy  : {accuracy:.3f}")
    print(f"  FalsePos  : {fpr:.3f}")

    print("\n" + "=" * 60)
    print("BY LANGUAGE")
    print("=" * 60)
    for lang, c in by_lang.items():
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        n = c["tp"] + c["fp"] + c["tn"] + c["fn"]
        acc = (c["tp"] + c["tn"]) / n if n else 0.0
        print(f"  {lang:9s}  n={n:3d}  P={p:.2f}  R={r:.2f}  F1={f:.2f}  Acc={acc:.2f}")

    print("\n" + "=" * 60)
    print("BY CATEGORY (accuracy)")
    print("=" * 60)
    for cat, c in sorted(by_cat.items()):
        acc = c["correct"] / c["total"] if c["total"] else 0.0
        print(f"  {cat:18s}  {c['correct']:2d}/{c['total']:2d}  = {acc:.2f}")

    print("\n" + "=" * 60)
    print("MISCLASSIFICATIONS (FP / FN only)")
    print("=" * 60)
    any_err = False
    for bucket, lang, cat, fired, snippet in confusion_rows:
        if bucket in ("FP", "FN"):
            any_err = True
            print(f"  [{bucket}] ({lang}/{cat}) fired={fired:10s} | {snippet}")
    if not any_err:
        print("  none - every prompt classified correctly")

    print()


if __name__ == "__main__":
    main()
