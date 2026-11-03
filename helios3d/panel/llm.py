"""The panel's LLM client. Injectable: the runner takes any object with a
``generate`` method of this shape, and tests pass a stub.

OllamaClient posts to ``<HELIOS3D_OLLAMA_URL>/api/generate`` with JSON
output, a fixed seed and temperature from the settings, and a timeout
below live's 400 s. It never reads OLLAMA_HOST or config.py: the base URL
comes only from the panel settings, which have no default.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None
    done_reason: str | None = None


class LLMCallError(RuntimeError):
    """One call failed (timeout, HTTP error, bad body). The decision goes on
    with that role marked failed."""


class LLMUnavailable(RuntimeError):
    """The server can't be reached. The run stops and the cursor stays."""


class LLMClient(Protocol):
    def generate(self, model: str, prompt: str, *, num_predict: int, num_ctx: int,
                 temperature: float, seed: int, timeout_s: float) -> LLMResult: ...


class OllamaClient:
    def __init__(self, base_url: str):
        if not base_url:
            raise ValueError("OllamaClient needs an explicit base URL (HELIOS3D_OLLAMA_URL)")
        self.base_url = base_url.rstrip("/")

    def generate(self, model, prompt, *, num_predict, num_ctx, temperature, seed, timeout_s):
        import requests
        payload = {"model": model, "prompt": prompt, "stream": False, "format": "json",
                   "options": {"num_predict": int(num_predict), "num_ctx": int(num_ctx),
                               "temperature": float(temperature), "seed": int(seed)}}
        t0 = time.monotonic()
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout_s)
            resp.raise_for_status()
            body = resp.json()
        except requests.exceptions.ConnectionError as e:
            raise LLMUnavailable(f"panel Ollama unreachable: {type(e).__name__}") from None
        except requests.exceptions.Timeout:
            raise LLMCallError(f"timeout after {timeout_s:.0f} s") from None
        except (requests.exceptions.RequestException, ValueError) as e:
            raise LLMCallError(call_error_text(e)) from None
        return LLMResult(text=body.get("response", "") or "",
                         prompt_tokens=body.get("prompt_eval_count"),
                         output_tokens=body.get("eval_count"),
                         latency_s=round(time.monotonic() - t0, 3),
                         done_reason=body.get("done_reason"))


# scheme://user:pass@ -> scheme:// (anything before '@' in the authority).
_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]*@")
ERROR_TEXT_CAP = 120


def strip_userinfo(text: str) -> str:
    """Text with the user:password part of every URL removed."""
    return _USERINFO_RE.sub(r"\1", text)


def call_error_text(e: BaseException) -> str:
    """What a failed call stores: the exception type, the HTTP status when
    there is a response, and the message with URL userinfo removed (before
    the cap, so a cut cannot leave half a credential)."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    head = type(e).__name__ + (f" HTTP {status}" if status is not None else "")
    return f"{head}: {strip_userinfo(str(e))[:ERROR_TEXT_CAP]}"


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def extract_json(text: str) -> dict:
    """The first JSON object in a reply (code fences and leading prose
    allowed). Raises ValueError when there is none."""
    if not isinstance(text, str):
        raise ValueError("reply is not text")
    s = _FENCE_RE.sub("", text.strip())
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    start = s.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(s[start:])
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj
