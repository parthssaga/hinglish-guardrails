"""
Language identification.

Decides whether an incoming message is English, Hindi (Devanagari),
or Hinglish (Roman-script Hindi, possibly mixed with English). The
pipeline uses this to decide whether transliteration is needed and to
tag every logged event with its language, which is what makes the
English-vs-Hinglish evaluation possible later.

Primary path: a small XLM-RoBERTa language classifier.
Fallback path: a script + wordlist heuristic that needs no model.
"""

from __future__ import annotations

import re

# A compact set of very common Roman-Hindi tokens. Presence of these in
# otherwise-Latin text is a strong signal of Hinglish. This is deliberately
# small; it only needs to catch the frequent function words.
_ROMAN_HINDI_MARKERS = {
    "hai", " hai", "nahi", "nahin", "kya", "kyun", "kyu", "mujhe", "tujhe",
    "aap", "tum", "mera", "tera", "humko", "hamko", "karo", "karna", "kar",
    "raha", "rahi", "rahe", "bhai", "yaar", "acha", "accha", "theek", "thik",
    "bhool", "jao", "matlab", "sirf", "bata", "batao", "chahiye", "chaiye",
    "kaise", "kaisa", "kaisi", "abhi", "phir", "bhi", "lekin", "magar",
    "haan", " haan", "wala", "wali", "kuch", "sab", "bohot", "bahut",
}

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
WORD = re.compile(r"[A-Za-z]+")


class LanguageIdentifier:
    def __init__(self, config):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            from config import MODELS

            name = MODELS["language_id"]
            self._tokenizer = AutoTokenizer.from_pretrained(name)
            self._model = AutoModelForSequenceClassification.from_pretrained(name)
            self._model.eval()
        except Exception:
            # No model available; we'll use the heuristic in `identify`.
            self._model = None
        self._loaded = True

    # -- heuristic fallback ------------------------------------------------
    @staticmethod
    def _heuristic(text: str) -> tuple[str, float]:
        if DEVANAGARI.search(text):
            return "hi", 0.95  # native-script Hindi

        lowered = " " + text.lower() + " "
        hits = sum(1 for m in _ROMAN_HINDI_MARKERS if m in lowered)
        n_words = max(1, len(WORD.findall(text)))
        ratio = hits / n_words

        if hits >= 1 and ratio > 0.08:
            return "hinglish", min(0.9, 0.5 + ratio)
        return "en", 0.8

    # -- public API --------------------------------------------------------
    def identify(self, text: str) -> tuple[str, float]:
        """
        Return (language_tag, confidence) where language_tag is one of
        'en', 'hi', or 'hinglish'.
        """
        if not self._loaded:
            self.load()

        # Devanagari is unambiguous; short-circuit regardless of model.
        if DEVANAGARI.search(text):
            return "hi", 0.97

        # The neural model distinguishes en/hi well but is not trained on
        # Roman-Hindi specifically, so we still run the Hinglish heuristic
        # and let it override an "en" verdict when markers are present.
        heur_lang, heur_conf = self._heuristic(text)
        if heur_lang == "hinglish":
            return heur_lang, heur_conf

        if self._model is not None:
            try:
                import torch

                inputs = self._tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=256
                )
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                idx = int(torch.argmax(probs))
                label = self._model.config.id2label[idx].lower()
                conf = float(probs[idx])
                # Map the model's label space to ours.
                if label.startswith("hi"):
                    return "hi", conf
                return "en", conf
            except Exception:
                pass

        return heur_lang, heur_conf
