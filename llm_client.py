"""
Provider-agnostic LLM client with Dual-Provider Resilient Failover.

Supports Groq (primary), Gemini, OpenAI, Anthropic, DeepSeek.
Automatically falls back to LLM_API_KEY_BACKUP (e.g. Gemini) upon 429/503/errors.
"""

from __future__ import annotations
import os
import json
import logging
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

LLM_API_KEY_BACKUP = os.environ.get("LLM_API_KEY_BACKUP", "")
LLM_PROVIDER_BACKUP = os.environ.get("LLM_PROVIDER_BACKUP", "")
LLM_MODEL_BACKUP = os.environ.get("LLM_MODEL_BACKUP", "")

# Default models per provider
_DEFAULT_MODELS = {
    "groq": "qwen/qwen3.8-27b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-3.5-flash-lite",
    "deepseek": "deepseek-chat",
}

# API endpoints
_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
}

_TIMEOUT = 12  # seconds — leaves safe buffer within tick budget


def _get_primary_config() -> tuple[str, str, str]:
    provider = os.environ.get("LLM_PROVIDER", LLM_PROVIDER).lower()
    api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
    model = os.environ.get("LLM_MODEL") or LLM_MODEL or _DEFAULT_MODELS.get(provider, "qwen/qwen3.8-27b")
    return provider, api_key, model


def _resolve_backup_config() -> tuple[str, str, str] | None:
    backup_key = os.environ.get("LLM_API_KEY_BACKUP", "").strip()
    if not backup_key:
        return None

    provider = os.environ.get("LLM_PROVIDER_BACKUP", "").strip().lower()
    if not provider:
        if backup_key.startswith("AQ.") or backup_key.startswith("AIzaSy"):
            provider = "gemini"
        elif backup_key.startswith("gsk_"):
            provider = "groq"
        elif backup_key.startswith("sk-"):
            provider = "openai"
        else:
            provider = "gemini"

    model = os.environ.get("LLM_MODEL_BACKUP", "").strip()
    if not model:
        if provider == "gemini":
            model = "gemini-3.5-flash-lite"
        elif provider == "groq":
            model = "qwen/qwen3.8-27b"
        elif provider == "openai":
            model = "gpt-4o-mini"
        else:
            model = _DEFAULT_MODELS.get(provider, "gemini-3.5-flash-lite")

    return provider, backup_key, model


def complete(prompt: str, system: str | None = None, temperature: float = 0,
             max_tokens: int = 800) -> str:
    """
    Send a chat completion request.
    If the primary provider hits a rate limit (429) or error, seamlessly falls back
    to the secondary provider (e.g. Gemini) without losing turn budget.
    """
    primary_prov, primary_key, primary_model = _get_primary_config()
    backup_cfg = _resolve_backup_config()

    try:
        return _execute_call(
            provider=primary_prov,
            api_key=primary_key,
            model=primary_model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            allow_retry=False if backup_cfg else True,
        )
    except Exception as e:
        if backup_cfg:
            b_prov, b_key, b_model = backup_cfg
            logger.warning(
                f"Primary LLM ({primary_prov}/{primary_model}) encountered error: {e}. "
                f"Failing over to backup LLM ({b_prov}/{b_model})..."
            )
            try:
                return _execute_call(
                    provider=b_prov,
                    api_key=b_key,
                    model=b_model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allow_retry=False,
                )
            except Exception as b_err:
                logger.error(f"Backup LLM failover ({b_prov}) also failed: {b_err}")
                raise b_err
        raise e


def _execute_call(provider: str, api_key: str, model: str,
                  prompt: str, system: str | None,
                  temperature: float, max_tokens: int,
                  allow_retry: bool = False) -> str:
    provider = provider.lower()
    if provider == "gemini":
        return _complete_gemini(prompt, system, temperature, max_tokens, api_key, model)
    elif provider == "anthropic":
        return _complete_anthropic(prompt, system, temperature, max_tokens, api_key, model)
    else:
        return _complete_openai_compat(prompt, system, temperature, max_tokens, provider, api_key, model, allow_retry)


def _complete_openai_compat(prompt: str, system: str | None,
                             temperature: float, max_tokens: int,
                             provider: str, api_key: str, model: str,
                             allow_retry: bool = False) -> str:
    """Works for Groq, OpenAI, DeepSeek — all OpenAI-compatible APIs."""
    import time
    endpoint = _ENDPOINTS.get(provider, _ENDPOINTS["groq"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    attempts = 2 if allow_retry else 1
    last_exc = None
    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.post(endpoint, json=body, headers=headers)
                if resp.status_code == 429 and allow_retry and attempt < attempts - 1:
                    retry_after = float(resp.headers.get("retry-after", 1.5))
                    time.sleep(min(retry_after, 2.0))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            last_exc = e
            if e.response.status_code == 429 and allow_retry and attempt < attempts - 1:
                time.sleep(1.5)
                continue
            raise
        except Exception as e:
            last_exc = e
            if allow_retry and attempt < attempts - 1:
                time.sleep(0.5)
                continue
            raise

    raise last_exc or RuntimeError(f"{provider} request failed after retries")


def _complete_anthropic(prompt: str, system: str | None,
                         temperature: float, max_tokens: int,
                         api_key: str, model: str) -> str:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]


def _complete_gemini(prompt: str, system: str | None,
                      temperature: float, max_tokens: int,
                      api_key: str, model: str) -> str:
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(url, json=body, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError(f"Gemini returned empty parts: {data}")
        return parts[0].get("text", "")

