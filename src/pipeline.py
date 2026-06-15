"""
The guardrail pipeline.

This is the orchestrator. For each user message it:

  1. identifies the language (en / hi / hinglish)
  2. runs the four input-side guardrails
        - PII first, so its redacted text feeds the others and the logs
        - toxicity, injection, jailbreak
  3. if anything fired (and block_on_input_flag is set) -> stop, log, return
  4. otherwise calls the LLM
  5. runs the two output-side guardrails on the response
  6. logs the whole trace to SQLite and returns a structured result

The return value is a dict carrying every guardrail result, so the chat
UI and the evaluation harness both get the full picture, not just a
yes/no.
"""

from __future__ import annotations

import time
import uuid

from config import PipelineConfig, DB_PATH
from src.preprocessing.language_id import LanguageIdentifier
from src.preprocessing.transliterate import Transliterator
from src.guardrails.toxicity import ToxicityGuardrail
from src.guardrails.pii import PIIGuardrail
from src.guardrails.injection import InjectionGuardrail
from src.guardrails.jailbreak import JailbreakGuardrail
from src.guardrails.output_filter import OutputFilterGuardrail
from src.guardrails.hallucination import HallucinationGuardrail
from src.llm import LLMClient
from src.logging_db import EventLog


class GuardrailPipeline:
    def __init__(self, config: PipelineConfig | None = None, db_path: str | None = None):
        self.config = config or PipelineConfig()

        # preprocessing
        self.lang_id = LanguageIdentifier(self.config)
        self.translit = Transliterator(self.config)

        # input guardrails
        self.pii = PIIGuardrail(self.config)
        self.toxicity = ToxicityGuardrail(self.config)
        self.injection = InjectionGuardrail(self.config)
        # jailbreak shares the transliterator so it can normalise Hinglish
        self.jailbreak = JailbreakGuardrail(self.config, transliterator=self.translit)

        # output guardrails
        self.output_filter = OutputFilterGuardrail(self.config)
        self.hallucination = HallucinationGuardrail(self.config)

        # llm + logging
        self.llm = LLMClient(self.config)
        self.log = EventLog(db_path or DB_PATH)

    # -- input side --------------------------------------------------------
    def _run_input_guardrails(self, text: str, language: str = "unknown"):
        """Returns (results, working_text, blocked_by)."""
        results = []
        working_text = text

        # PII first: redact, then everything downstream sees redacted text.
        if self.config.enable_pii:
            r = self.pii.check(working_text)
            results.append(r)
            if r.sanitized_text is not None:
                working_text = r.sanitized_text

        if self.config.enable_toxicity:
            results.append(self.toxicity.check(working_text))
        if self.config.enable_injection:
            # Pass the detected language so DeBERTa is skipped for Hindi,
            # where it produces false positives on Devanagari text.
            results.append(self.injection.check(working_text, language=language))
        if self.config.enable_jailbreak:
            results.append(self.jailbreak.check(working_text))

        # Decide whether to block. PII alone redacts but does not block;
        # the others block when triggered.
        blocked_by = None
        for r in results:
            if r.name == "pii":
                continue
            if r.triggered:
                blocked_by = r.name
                break

        return results, working_text, blocked_by

    # -- output side -------------------------------------------------------
    def _run_output_guardrails(
        self,
        response_text: str,
        avg_logprob,
        user_text: str | None = None,
        source: str | None = None,
    ):
        results = []
        flagged_by = None

        if self.config.enable_output_filter:
            r = self.output_filter.check(response_text)
            results.append(r)
            if r.triggered:
                # Report the most specific category that fired so refusal
                # messages and logs can explain WHY the output was blocked.
                fired = r.metadata.get("fired_categories", [])
                flagged_by = f"output_filter/{fired[0]}" if fired else r.name

        if self.config.enable_hallucination:
            if source:
                r = self.hallucination.check_grounded(response_text, source)
            else:
                r = self.hallucination.check_with_logprobs(
                    response_text, avg_logprob, user_query=user_text
                )
            results.append(r)
            # hallucination only flags; it does not block

        return results, flagged_by

    # -- public entry (guardrails only, no LLM) ---------------------------
    def check_input(self, text: str) -> dict:
        """Run input guardrails only; skip the LLM.  Does not write to the log.

        Returns the same shape as ``process()`` but without ``response`` or
        ``output_results``.  Used by the REST API ``POST /check`` endpoint.
        """
        t0 = time.perf_counter()
        if self.config.enable_language_id:
            language, lang_conf = self.lang_id.identify(text)
        else:
            language, lang_conf = "unknown", 0.0

        input_results, _, blocked_by = self._run_input_guardrails(
            text, language=language
        )
        total_ms = (time.perf_counter() - t0) * 1000
        return {
            "language": language,
            "lang_conf": round(lang_conf, 3),
            "blocked": blocked_by is not None,
            "blocked_by": blocked_by,
            "input_results": [r.as_dict() for r in input_results],
            "total_ms": round(total_ms, 1),
        }

    # -- public entry (full pipeline) --------------------------------------
    def process(
        self,
        user_message: str,
        conversation_id: str | None = None,
        history: list[dict] | None = None,
        source: str | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        conversation_id = conversation_id or str(uuid.uuid4())[:8]

        # 1. language
        if self.config.enable_language_id:
            language, lang_conf = self.lang_id.identify(user_message)
        else:
            language, lang_conf = "unknown", 0.0

        # 2-3. input guardrails
        input_results, working_text, blocked_by = self._run_input_guardrails(
            user_message, language=language
        )

        redacted_input = working_text  # safe to store (PII already stripped)

        if blocked_by and self.config.block_on_input_flag:
            total_ms = (time.perf_counter() - t0) * 1000
            self.log.record(
                conversation=conversation_id,
                language=language, lang_conf=lang_conf,
                user_input=redacted_input,
                final_action="blocked", blocked_by=blocked_by,
                response=None,
                input_results=[r.as_dict() for r in input_results],
                output_results=[],
                total_ms=total_ms,
            )
            return {
                "conversation": conversation_id,
                "language": language,
                "blocked": True,
                "blocked_by": blocked_by,
                "response": self._refusal_message(blocked_by, language),
                "input_results": [r.as_dict() for r in input_results],
                "output_results": [],
                "total_ms": round(total_ms, 1),
            }

        # 4. LLM (send the redacted text, never the raw PII)
        llm_resp = self.llm.generate(working_text, history=history)
        if llm_resp.error:
            total_ms = (time.perf_counter() - t0) * 1000
            return {
                "conversation": conversation_id,
                "language": language,
                "blocked": False,
                "blocked_by": None,
                "response": f"[LLM error] {llm_resp.error}",
                "input_results": [r.as_dict() for r in input_results],
                "output_results": [],
                "total_ms": round(total_ms, 1),
                "error": llm_resp.error,
            }

        # 5. output guardrails
        output_results, flagged_by = self._run_output_guardrails(
            llm_resp.text, llm_resp.avg_logprob,
            user_text=working_text, source=source,
        )

        # If the output filter flagged the response as unsafe, replace it.
        final_response = llm_resp.text
        final_action = "allowed"
        out_blocked_by = None
        if flagged_by:
            final_response = self._refusal_message(flagged_by, language)
            final_action = "blocked"
            out_blocked_by = flagged_by

        total_ms = (time.perf_counter() - t0) * 1000
        self.log.record(
            conversation=conversation_id,
            language=language, lang_conf=lang_conf,
            user_input=redacted_input,
            final_action=final_action,
            blocked_by=out_blocked_by,
            response=final_response if final_action == "allowed" else None,
            input_results=[r.as_dict() for r in input_results],
            output_results=[r.as_dict() for r in output_results],
            total_ms=total_ms,
        )

        return {
            "conversation": conversation_id,
            "language": language,
            "blocked": final_action == "blocked",
            "blocked_by": out_blocked_by,
            "response": final_response,
            "input_results": [r.as_dict() for r in input_results],
            "output_results": [r.as_dict() for r in output_results],
            "total_ms": round(total_ms, 1),
        }

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _refusal_message(reason: str, language: str) -> str:
        msgs = {
            # input-side
            "toxicity":  "I can't engage with messages that contain abusive or hateful language.",
            "injection":  "That request looks like an attempt to change my instructions, so I won't act on it.",
            "jailbreak":  "I can't help with that request.",
            # output-side — generic fallback
            "output_filter": "I generated a response that didn't meet safety standards, so I've withheld it.",
            # output-side — per category (flagged_by = "output_filter/<category>")
            "output_filter/toxic":
                "My response contained harmful language, so I've withheld it.",
            "output_filter/system_prompt_leak":
                "I can't share my internal instructions.",
            "output_filter/unsafe_compliance":
                "My response to that request wasn't safe to share, so I've withheld it.",
            "output_filter/pii_in_output":
                "My response contained personal information, which I've withheld to protect privacy.",
        }
        base = msgs.get(reason, "I can't help with that request.")
        if language == "hinglish":
            base += " (Yeh response safe nahi tha.)"
        return base
