"""
Evaluation harness.

INPUT MODE (default):
    Runs the input-side guardrails over the labelled set in
    data/test_prompts.json and reports precision, recall, F1, accuracy, and
    false-positive rate, broken down by language (English vs Hinglish) and by
    attack category. This directly produces the pipeline-level numbers the
    research paper needs.

OUTPUT MODE (--output-mode):
    Evaluates the output-side guardrails (OutputFilterGuardrail +
    HallucinationGuardrail grounding check) over output_benchmark.json.
    Reports per-category P/R/F1 (treating each harm category as a binary
    classifier), per-language accuracy, and grounding check P/R/F1.

Both modes run without a running LLM and without network access — only the
guardrail models (or their rule-based fallbacks) are required.

Usage:
    python evaluate.py
    python evaluate.py --data data/test_prompts.json
    python evaluate.py --output-mode
    python evaluate.py --output-mode --output-data output_benchmark.json
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
from src.guardrails.output_filter import OutputFilterGuardrail
from src.guardrails.hallucination import HallucinationGuardrail


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


# ---------------------------------------------------------------------------
# Input-side evaluation
# ---------------------------------------------------------------------------

def run_input_eval(data_path: str) -> None:
    with open(data_path, encoding="utf-8") as f:
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


# ---------------------------------------------------------------------------
# Output-side evaluation
# ---------------------------------------------------------------------------

_OUTPUT_CATEGORIES = ["toxic", "system_prompt_leak", "unsafe_compliance", "pii_in_output"]


def run_output_eval(data_path: str) -> None:
    with open(data_path, encoding="utf-8") as f:
        dataset = json.load(f)

    output_items = dataset.get("output_items", [])
    grounding_items = dataset.get("grounding_items", [])

    config = PipelineConfig()
    output_filter = OutputFilterGuardrail(config)
    hallucination = HallucinationGuardrail(config)

    print(f"Evaluating {len(output_items)} output items + {len(grounding_items)} grounding items...\n")

    # ------------------------------------------------------------------
    # Output filter evaluation
    # ------------------------------------------------------------------

    # Per-category confusion matrices (positive class = category fires)
    cat_counts: dict[str, dict[str, int]] = {
        cat: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for cat in _OUTPUT_CATEGORIES
    }
    # Per-language item-level accuracy
    lang_correct: dict[str, int] = defaultdict(int)
    lang_total: dict[str, int] = defaultdict(int)
    out_confusion: list[tuple] = []

    for item in output_items:
        text = item["text"]
        lang = item["language"]
        expected_cat = item["category"]   # "safe" | "toxic" | …

        result = output_filter.check(text)
        fired = set(result.metadata.get("fired_categories", []))

        # Per-category binary evaluation
        for cat in _OUTPUT_CATEGORIES:
            should_fire = (expected_cat == cat)
            did_fire = cat in fired
            if should_fire and did_fire:
                cat_counts[cat]["tp"] += 1
            elif should_fire and not did_fire:
                cat_counts[cat]["fn"] += 1
            elif not should_fire and did_fire:
                cat_counts[cat]["fp"] += 1
            else:
                cat_counts[cat]["tn"] += 1

        # Item-level correctness for language breakdown
        if expected_cat == "safe":
            correct = len(fired) == 0
        else:
            correct = expected_cat in fired

        lang_total[lang] += 1
        if correct:
            lang_correct[lang] += 1

        if not correct:
            bucket = "FN" if expected_cat != "safe" else "FP"
            out_confusion.append((bucket, lang, expected_cat, sorted(fired), text[:60]))

    # ------------------------------------------------------------------
    # Overall output-filter accuracy
    # ------------------------------------------------------------------
    total_out = len(output_items)
    total_correct = sum(lang_correct.values())
    overall_acc = total_correct / total_out if total_out else 0.0

    print("=" * 60)
    print("OUTPUT FILTER — OVERALL")
    print("=" * 60)
    print(f"  n={total_out}  Correct={total_correct}  Accuracy={overall_acc:.3f}")

    # ------------------------------------------------------------------
    # Per-category P/R/F1
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OUTPUT FILTER — PER CATEGORY (positive class = category fires)")
    print("=" * 60)
    for cat in _OUTPUT_CATEGORIES:
        c = cat_counts[cat]
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        n_pos = c["tp"] + c["fn"]
        n_neg = c["fp"] + c["tn"]
        print(
            f"  {cat:22s}  TP={c['tp']:3d} FP={c['fp']:3d} "
            f"TN={c['tn']:3d} FN={c['fn']:3d}  "
            f"P={p:.2f}  R={r:.2f}  F1={f:.2f}  "
            f"(pos={n_pos} neg={n_neg})"
        )

    # ------------------------------------------------------------------
    # Per-language output-filter accuracy
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OUTPUT FILTER — BY LANGUAGE")
    print("=" * 60)
    for lang in ("en", "hinglish", "hi"):
        n = lang_total.get(lang, 0)
        c = lang_correct.get(lang, 0)
        acc = c / n if n else 0.0
        print(f"  {lang:9s}  n={n:3d}  Correct={c:3d}  Acc={acc:.3f}")

    # ------------------------------------------------------------------
    # Output filter misclassifications
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OUTPUT FILTER — MISCLASSIFICATIONS")
    print("=" * 60)
    if out_confusion:
        for bucket, lang, cat, fired_cats, snippet in out_confusion[:40]:
            fired_str = ",".join(fired_cats) if fired_cats else "-"
            print(f"  [{bucket}] ({lang}/{cat:22s}) fired=[{fired_str:22s}] | {snippet}")
        if len(out_confusion) > 40:
            print(f"  … {len(out_confusion) - 40} more (truncated)")
    else:
        print("  none - every output classified correctly")

    # ------------------------------------------------------------------
    # Grounding evaluation
    # ------------------------------------------------------------------
    if not grounding_items:
        print("\n(No grounding items in dataset — skipping grounding evaluation)")
        print()
        return

    print("\n" + "=" * 60)
    print("GROUNDING CHECK (positive class = ungrounded)")
    print("=" * 60)

    g_tp = g_fp = g_tn = g_fn = 0
    g_lang_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    )
    g_confusion: list[tuple] = []
    grounding_unavailable = False

    for item in grounding_items:
        response = item["response"]
        source = item["source"]
        lang = item.get("language", "en")
        expected_ungrounded = item["expected"] == "ungrounded"

        result = hallucination.check_grounded(response, source)

        # If model is unavailable, all items return triggered=False
        if result.metadata.get("mode") == "grounded/unavailable":
            grounding_unavailable = True
            continue

        pred_ungrounded = result.triggered

        if expected_ungrounded and pred_ungrounded:
            g_tp += 1
            g_lang_counts[lang]["tp"] += 1
        elif expected_ungrounded and not pred_ungrounded:
            g_fn += 1
            g_lang_counts[lang]["fn"] += 1
            g_confusion.append(("FN", lang, response[:60], source[:60]))
        elif not expected_ungrounded and pred_ungrounded:
            g_fp += 1
            g_lang_counts[lang]["fp"] += 1
            g_confusion.append(("FP", lang, response[:60], source[:60]))
        else:
            g_tn += 1
            g_lang_counts[lang]["tn"] += 1

    if grounding_unavailable and (g_tp + g_fp + g_tn + g_fn) == 0:
        print("  Embedder model unavailable — grounding metrics cannot be computed.")
        print("  Install sentence-transformers and cache paraphrase-multilingual-MiniLM-L12-v2.")
        print()
        return

    if grounding_unavailable:
        print("  Note: some grounding items skipped (embedder unavailable).")

    g_total = g_tp + g_fp + g_tn + g_fn
    g_acc = (g_tp + g_tn) / g_total if g_total else 0.0
    g_p, g_r, g_f = prf(g_tp, g_fp, g_fn)

    print(f"  n={g_total}  TP={g_tp}  FP={g_fp}  TN={g_tn}  FN={g_fn}")
    print(f"  Precision : {g_p:.3f}")
    print(f"  Recall    : {g_r:.3f}")
    print(f"  F1        : {g_f:.3f}")
    print(f"  Accuracy  : {g_acc:.3f}")

    print("\n" + "=" * 60)
    print("GROUNDING CHECK — BY LANGUAGE")
    print("=" * 60)
    for lang in ("en", "hinglish", "hi"):
        c = g_lang_counts.get(lang, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        n = c["tp"] + c["fp"] + c["tn"] + c["fn"]
        if n == 0:
            continue
        acc = (c["tp"] + c["tn"]) / n
        gp, gr, gf = prf(c["tp"], c["fp"], c["fn"])
        print(
            f"  {lang:9s}  n={n:3d}  TP={c['tp']:2d} FP={c['fp']:2d} "
            f"TN={c['tn']:2d} FN={c['fn']:2d}  "
            f"P={gp:.2f}  R={gr:.2f}  F1={gf:.2f}  Acc={acc:.2f}"
        )

    if g_confusion:
        print("\n" + "=" * 60)
        print("GROUNDING MISCLASSIFICATIONS (FP / FN only)")
        print("=" * 60)
        for bucket, lang, resp_snip, src_snip in g_confusion[:20]:
            print(f"  [{bucket}] ({lang})")
            print(f"    resp: {resp_snip}")
            print(f"    src : {src_snip}")
        if len(g_confusion) > 20:
            print(f"  … {len(g_confusion) - 20} more (truncated)")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate input or output guardrails on a labelled benchmark."
    )
    ap.add_argument("--data", default="data/test_prompts.json",
                    help="Path to input benchmark JSON (default: data/test_prompts.json)")
    ap.add_argument("--output-mode", action="store_true",
                    help="Evaluate output-side guardrails (OutputFilterGuardrail + grounding)")
    ap.add_argument("--output-data", default="output_benchmark.json",
                    help="Path to output benchmark JSON (default: output_benchmark.json)")
    args = ap.parse_args()

    if args.output_mode:
        run_output_eval(args.output_data)
    else:
        run_input_eval(args.data)


if __name__ == "__main__":
    main()
