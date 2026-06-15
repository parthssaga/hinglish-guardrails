"""
Thin wrapper around a local Ollama model.

Ollama runs an open-source LLM (e.g. llama3.2, qwen2.5) entirely on your
own machine: no API key, no per-message cost, no internet needed once the
model is pulled. Isolating the call here means the rest of the system
never talks to Ollama directly, so swapping providers later touches only
this file.

Note on hallucination detection: Ollama's standard chat endpoint does not
return per-token logprobs, so `avg_logprob` is always None here and the
hallucination guardrail falls back to its heuristic mode. That is wired
to happen automatically.

Prerequisites (one-time, on your Mac):
    1. Install Ollama:           https://ollama.com/download
    2. Pull a small model:       ollama pull llama3.2
    3. Make sure it's running:   the Ollama app, or `ollama serve`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    text: str
    avg_logprob: Optional[float]  # always None for Ollama (no logprobs)
    raw: dict = None
    error: Optional[str] = None


class LLMClient:
    def __init__(self, config):
        self.config = config

    def generate(self, user_message: str, history: list[dict] | None = None) -> LLMResponse:
        from config import (
            OLLAMA_MODEL,
            OLLAMA_HOST,
            LLM_MAX_TOKENS,
            LLM_TEMPERATURE,
            SYSTEM_PROMPT,
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # Try the official python client first; fall back to a raw HTTP
        # POST so the code still works if the package isn't installed.
        try:
            import ollama

            client = ollama.Client(host=OLLAMA_HOST)
            resp = client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_MAX_TOKENS,
                },
            )
            text = resp.get("message", {}).get("content", "")
            return LLMResponse(text=text, avg_logprob=None, raw=dict(resp))

        except ImportError:
            # package not installed -> use plain HTTP via urllib (stdlib)
            return self._http_generate(
                messages, OLLAMA_HOST, OLLAMA_MODEL,
                LLM_TEMPERATURE, LLM_MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001
            hint = (
                " (Is Ollama running? Start the Ollama app or run "
                "`ollama serve`, and make sure you've run "
                f"`ollama pull {OLLAMA_MODEL}`.)"
            )
            return LLMResponse(text="", avg_logprob=None, error=str(exc) + hint)

    @staticmethod
    def _http_generate(messages, host, model, temperature, max_tokens) -> LLMResponse:
        import json
        import urllib.request
        import urllib.error

        url = host.rstrip("/") + "/api/chat"
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            text = data.get("message", {}).get("content", "")
            return LLMResponse(text=text, avg_logprob=None, raw=data)
        except urllib.error.URLError as exc:
            hint = (
                " (Could not reach Ollama at "
                f"{host}. Start the Ollama app or run `ollama serve`, "
                f"and run `ollama pull {model}` first.)"
            )
            return LLMResponse(text="", avg_logprob=None, error=str(exc) + hint)
        except Exception as exc:  # noqa: BLE001
            return LLMResponse(text="", avg_logprob=None, error=str(exc))
