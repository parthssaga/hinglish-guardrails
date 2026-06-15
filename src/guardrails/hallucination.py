"""
Hallucination flag (output side) — multi-signal composite scorer.

Upgraded from a single logprob/hedge check to a four-signal composite:

  logprob_confidence    — model's own token-level probability (primary signal
                          when Ollama provides logprobs; higher logprob_score
                          = more uncertain)
  hedge_density         — normalized rate of uncertainty markers per sentence
                          (EN + Hinglish + Devanagari phrase lists)
  numeric_overconfidence — specific statistics / percentages stated without
                          an uncertainty qualifier or source attribution
  self_contradiction    — absolute claims (always/never/all) co-occurring
                          with softeners (sometimes/can/may) across
                          different sentences on the same topic
  temporal_overreach    — assertive present-tense claims about inherently
                          time-sensitive facts ("the current X is …") without
                          a source or date qualifier

Any signal that exceeds its individual threshold is listed in
metadata["fired_signals"].  The overall score is a weighted composite of
all active signals, amplified by logprob uncertainty when available.

An optional NLI consistency check (cross-encoder/nli-deberta-v3-small)
can be enabled by passing user_query to check_with_logprobs; it degrades
gracefully to 0.0 when sentence-transformers is not installed or the model
is not cached.

This guardrail flags but does NOT block — see pipeline.py.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from src.guardrails.base import BaseGuardrail, GuardrailResult


# ---------------------------------------------------------------------------
# Signal 1 — hedge_density
# ---------------------------------------------------------------------------

_HEDGES_EN: frozenset[str] = frozenset({
    "i'm not sure", "i am not sure", "i think", "possibly", "it might be",
    "as far as i know", "i believe", "probably", "i'm not certain",
    "i'm not 100%", "i could be wrong", "i may be mistaken", "not sure if",
    "to my knowledge", "if i recall correctly", "i seem to recall",
    "may or may not", "i'm unsure", "i cannot confirm", "i'm not aware",
    "please verify", "please double-check", "please check this",
    "i'd suggest verifying", "don't quote me", "i'm not entirely sure",
    "uncertain", "i'm not confident", "take this with a grain of salt",
})

_HEDGES_HINGLISH: frozenset[str] = frozenset({
    "shayad", "lagta hai", "mujhe nahi pata", "mujhe yakeen nahi",
    "ho sakta hai", "meri jaankari ke mutabiq", "verify kar lena",
    "pakka nahi hoon", "pakka nahi", "mujhe pura yakeen nahi",
})

_HEDGES_DEVANAGARI: frozenset[str] = frozenset({
    "शायद", "लगता है", "मुझे नहीं पता", "मुझे यकीन नहीं",
    "हो सकता है", "मेरी जानकारी के मुताबिक", "सत्यापित करें",
    "पक्का नहीं हूँ", "पक्का नहीं", "मुझे पूरा यकीन नहीं",
})

_ALL_HEDGES: frozenset[str] = _HEDGES_EN | _HEDGES_HINGLISH | _HEDGES_DEVANAGARI


# ---------------------------------------------------------------------------
# Signal 2 — numeric_overconfidence
# ---------------------------------------------------------------------------

_NUMERIC_STAT_RE = re.compile(
    r"(\b\d+\.?\d*\s*%)"
    r"|\b\d{1,3}(?:,\d{3})+\s+(?:people|users|cases|deaths|dollars|years|times)\b"
    r"|\b\d{5,}\s+(?:people|users|cases|deaths|dollars|years|times)\b",
    re.IGNORECASE,
)

_NUMERIC_QUALIFIER_RE = re.compile(
    r"\b(approximately|about|around|roughly|nearly|almost|estimated|"
    r"according to|sources? say|"
    r"(?:a |the )?study (?:by|from|in|published)\b|"
    r"research (?:shows?|suggests?) that\b|"
    r"shayad|lagbhag|taqreeban|लगभग|तकरीबन|अनुमानित)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Signal 3 — self_contradiction
# ---------------------------------------------------------------------------

_ABSOLUTE_RE = re.compile(
    r"\b(always|never|all|none|every|no one|everyone|nobody|"
    r"impossible|definitely|certainly|absolutely|guaranteed|100%|"
    r"without exception|under no circumstances|invariably)\b",
    re.IGNORECASE,
)

_SOFTENER_RE = re.compile(
    r"\b(sometimes|occasionally|in some cases|can|may|could|might|"
    r"possible|varies?|certain|some|partial(?:ly)?|it depends?|"
    r"generally|usually|often|rarely|not always|exceptions?)\b",
    re.IGNORECASE,
)

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "it", "in", "of", "to", "and", "or", "for",
    "with", "that", "this", "i", "you", "be", "are", "was", "were", "have",
    "has", "had", "but", "not", "so", "do", "did", "on", "at", "by", "from",
    "up", "about", "into", "through", "during", "before", "after", "than",
    "then", "also", "however", "though", "although", "because", "since", "if",
    # signal words themselves don't count as content-word overlap
    "always", "never", "none", "every", "sometimes", "occasionally",
    "can", "may", "could", "might", "generally", "usually", "often",
})


# ---------------------------------------------------------------------------
# Signal 4 — temporal_overreach
# ---------------------------------------------------------------------------

_TEMPORAL_OVERREACH_RE = re.compile(
    r"\b(current(?:ly)?|latest|newest|most recent|"
    r"right now|as of (?:now|today|\d{4})|"
    r"today|this (?:year|month|week)|"
    r"aaj kal|abhi|अभी|आजकल|फिलहाल)\b"
    r".{0,80}"
    r"\b(is|are|has|have|will be|was|'s)\b",
    re.IGNORECASE | re.DOTALL,
)

_TEMPORAL_QUALIFIER_RE = re.compile(
    r"\b(as of \d{4}|according to|last updated|published|released|"
    r"announced|reported|when (?:I was|my knowledge))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Per-signal thresholds and composite weights
# ---------------------------------------------------------------------------

_SIGNAL_THRESHOLDS: dict[str, float] = {
    "hedge_density":          0.50,
    "numeric_overconfidence": 0.55,
    "self_contradiction":     0.60,
    "temporal_overreach":     0.55,
    "nli_consistency":        0.60,
}

_SIGNAL_WEIGHTS: dict[str, float] = {
    "hedge_density":          0.40,
    "numeric_overconfidence": 0.70,
    "self_contradiction":     0.90,
    "temporal_overreach":     0.50,
    "nli_consistency":        0.80,
}


class HallucinationGuardrail(BaseGuardrail):
    """Multi-signal hallucination flag.

    This is a *flag*, not a hard block — hallucination detection is
    inherently probabilistic.  The guardrail surfaces risk for downstream
    inspection without suppressing the model's response.
    """

    name = "hallucination"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._nli = None
        self._embedder = None

    def load(self) -> None:
        if self._ready:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")
        except Exception:
            self._nli = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._embedder = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception:
            self._embedder = None
        self._ready = True

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def check_with_logprobs(
        self,
        text: str,
        avg_logprob,
        user_query: str | None = None,
    ) -> GuardrailResult:
        """Preferred entry point.

        avg_logprob  — mean per-token logprob from Ollama (negative float
                       or None if unavailable).
        user_query   — optional original user message; enables NLI
                       consistency check when the cross-encoder is loaded.
        """
        start = __import__("time").perf_counter()

        # "hallucination_confidence" = max acceptable model confidence;
        # fire when composite uncertainty score >= 1 - threshold.
        threshold = self.config.thresholds["hallucination_confidence"]
        trigger_at = 1.0 - threshold  # e.g. 0.45 → trigger at score >= 0.55

        # Heuristic signals (always run)
        signal_scores: dict[str, float] = {
            "hedge_density":          self._signal_hedge_density(text),
            "numeric_overconfidence": self._signal_numeric_overconfidence(text),
            "self_contradiction":     self._signal_self_contradiction(text),
            "temporal_overreach":     self._signal_temporal_overreach(text),
        }
        if self._nli is not None and user_query:
            signal_scores["nli_consistency"] = self._signal_nli_consistency(
                text, user_query
            )

        # Weighted composite
        total_w = sum(_SIGNAL_WEIGHTS[k] for k in signal_scores)
        composite = (
            sum(v * _SIGNAL_WEIGHTS[k] for k, v in signal_scores.items()) / total_w
            if total_w > 0 else 0.0
        )

        confidence: float | None = None
        backend = "heuristic"

        if avg_logprob is not None:
            confidence = math.exp(avg_logprob)
            logprob_score = 1.0 - confidence
            # Heuristic contribution is gated by model uncertainty: low
            # logprob_score (high confidence) dampens the composite so that
            # heuristic false-positives don't override a confident model.
            final_score = max(logprob_score, composite * min(1.0, logprob_score * 2))
            backend = "logprob+heuristic"
        else:
            final_score = composite

        fired_signals = [
            k for k, v in signal_scores.items()
            if v >= _SIGNAL_THRESHOLDS.get(k, 0.50)
        ]

        triggered = final_score >= trigger_at
        label = ", ".join(fired_signals) if fired_signals else "low model confidence"
        reason = (
            f"hallucination risk — signals: {label} (score {final_score:.2f})"
            if triggered
            else f"no strong hallucination indicators (score {final_score:.2f})"
        )

        meta: dict[str, Any] = {
            "backend": backend,
            "fired_signals": fired_signals,
            "signal_scores": {k: round(v, 3) for k, v in signal_scores.items()},
            "composite_score": round(composite, 3),
        }
        if confidence is not None:
            meta["confidence"] = round(confidence, 3)

        res = GuardrailResult(
            name=self.name,
            triggered=triggered,
            score=round(final_score, 3),
            reason=reason,
            metadata=meta,
        )
        res.elapsed_ms = (__import__("time").perf_counter() - start) * 1000
        return res

    def _check(self, text: str) -> GuardrailResult:
        return self.check_with_logprobs(text, None)

    # ------------------------------------------------------------------
    # Grounding-based check (preferred when source text is available)
    # ------------------------------------------------------------------

    def check_grounded(self, response: str, source_text: str) -> GuardrailResult:
        """Verify whether the response's claims are supported by source_text.

        Splits both texts into sentences, encodes them with
        paraphrase-multilingual-MiniLM-L12-v2 (supports EN/Hindi/Hinglish),
        and computes cosine similarity between each response sentence and the
        closest source sentence.  Sentences whose max-similarity falls below
        the grounding threshold are flagged as potentially unsupported.

        result.reason always starts with "grounded check:" so callers can
        distinguish this path from the confidence-heuristic path.

        Degrades gracefully: if sentence-transformers is not installed or the
        model is not cached, returns triggered=False with
        metadata["mode"] == "grounded/unavailable".
        """
        self.load()
        start = __import__("time").perf_counter()

        if self._embedder is None:
            res = GuardrailResult(
                name=self.name,
                triggered=False,
                score=0.0,
                reason=(
                    "grounded mode unavailable — install sentence-transformers "
                    "and cache paraphrase-multilingual-MiniLM-L12-v2"
                ),
                metadata={"mode": "grounded/unavailable"},
            )
            res.elapsed_ms = (__import__("time").perf_counter() - start) * 1000
            return res

        threshold = self.config.thresholds.get("hallucination_grounding", 0.35)

        resp_sents = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", response)     if s.strip()]
        src_sents  = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", source_text)  if s.strip()]

        if not resp_sents or not src_sents:
            res = GuardrailResult(
                name=self.name,
                triggered=False,
                score=0.0,
                reason="grounded check: no sentences to compare",
                metadata={
                    "mode": "grounded",
                    "sentences": [],
                    "unsupported_count": 0,
                    "total_sentences": 0,
                },
            )
            res.elapsed_ms = (__import__("time").perf_counter() - start) * 1000
            return res

        try:
            emb = self._embedder.encode(
                resp_sents + src_sents,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            resp_emb = emb[:len(resp_sents)]
            src_emb  = emb[len(resp_sents):]

            # Cosine similarity matrix (n_resp × n_src)
            r_norm = resp_emb / (np.linalg.norm(resp_emb, axis=1, keepdims=True) + 1e-9)
            s_norm = src_emb  / (np.linalg.norm(src_emb,  axis=1, keepdims=True) + 1e-9)
            sim = r_norm @ s_norm.T

            per_sentence = [
                {
                    "text": sent,
                    "max_similarity": round(float(sim[i].max()), 3),
                    "supported": float(sim[i].max()) >= threshold,
                }
                for i, sent in enumerate(resp_sents)
            ]

            n_unsup      = sum(1 for s in per_sentence if not s["supported"])
            n_total      = len(per_sentence)
            mean_max_sim = float(sim.max(axis=1).mean())
            score        = round(max(0.0, 1.0 - mean_max_sim), 3)
            triggered    = n_unsup > 0

            reason = (
                f"grounded check: {n_unsup}/{n_total} sentences unsupported "
                f"(score {score:.2f})"
                if triggered
                else f"grounded check: all {n_total} sentences appear supported "
                f"(score {score:.2f})"
            )
            meta: dict[str, Any] = {
                "mode": "grounded",
                "model": "paraphrase-multilingual-MiniLM-L12-v2",
                "grounding_threshold": threshold,
                "sentences": per_sentence,
                "unsupported_count": n_unsup,
                "total_sentences": n_total,
                "unsupported_fraction": round(n_unsup / n_total, 3) if n_total else 0.0,
                "mean_max_similarity": round(mean_max_sim, 3),
            }
        except Exception as exc:
            triggered = False
            score     = 0.0
            reason    = f"grounded check error: {exc}"
            meta      = {"mode": "grounded/error", "error": str(exc)}

        res = GuardrailResult(
            name=self.name,
            triggered=triggered,
            score=score,
            reason=reason,
            metadata=meta,
        )
        res.elapsed_ms = (__import__("time").perf_counter() - start) * 1000
        return res

    # ------------------------------------------------------------------
    # Individual signal scorers — each returns float in [0, 1]
    # ------------------------------------------------------------------

    def _signal_hedge_density(self, text: str) -> float:
        """Normalized rate of uncertainty markers; requires ≥2 hits to score."""
        low = text.lower()
        hits = sum(1 for h in _ALL_HEDGES if h in low)
        if hits < 2:
            return 0.0
        sentences = max(1, len(re.split(r"[.!?।]", text)))
        return min(1.0, (hits / sentences) * 1.5)

    def _signal_numeric_overconfidence(self, text: str) -> float:
        """Specific statistics without an uncertainty qualifier or source."""
        n = len(list(_NUMERIC_STAT_RE.finditer(text)))
        if n == 0:
            return 0.0
        base = min(0.90, 0.40 + 0.15 * n)
        has_qualifier = bool(_NUMERIC_QUALIFIER_RE.search(text))
        return base * (0.35 if has_qualifier else 1.0)

    def _signal_self_contradiction(self, text: str) -> float:
        """Absolute claim coexisting with a softener across different sentences."""
        sentences = [s for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]
        if len(sentences) < 2:
            return 0.0

        abs_idxs = {i for i, s in enumerate(sentences) if _ABSOLUTE_RE.search(s)}
        soft_idxs = {i for i, s in enumerate(sentences) if _SOFTENER_RE.search(s)}

        if not abs_idxs or not soft_idxs:
            return 0.0

        # Both signals confined to the exact same sentences = nuanced writing,
        # not contradiction (e.g. "It always rains, though sometimes it's sunny.")
        if abs_idxs == soft_idxs:
            return 0.0

        # Content-word overlap confirms the same topic is being contradicted
        def _content_words(idx_set: set[int]) -> set[str]:
            words: set[str] = set()
            for i in idx_set:
                words.update(
                    w for w in re.findall(r"\b[a-z]+\b", sentences[i].lower())
                    if w not in _STOPWORDS and len(w) > 3
                )
            return words

        shared = _content_words(abs_idxs) & _content_words(soft_idxs)
        if len(shared) >= 2:
            return 0.70
        if len(shared) >= 1:
            return 0.40
        return 0.0

    def _signal_temporal_overreach(self, text: str) -> float:
        """Assertive present-tense claim about a time-sensitive fact."""
        matches = _TEMPORAL_OVERREACH_RE.findall(text)
        if not matches:
            return 0.0
        if bool(_TEMPORAL_QUALIFIER_RE.search(text)):
            return 0.0
        return min(0.80, 0.40 + 0.15 * len(matches))

    def _signal_nli_consistency(self, text: str, user_query: str) -> float:
        """Cross-encoder NLI: returns contradiction probability (0 = consistent)."""
        try:
            scores = self._nli.predict([(user_query, text)])
            probs = scores[0]
            # cross-encoder/nli-deberta-v3-small: [contradiction, entailment, neutral]
            return float(probs[0])
        except Exception:
            return 0.0
