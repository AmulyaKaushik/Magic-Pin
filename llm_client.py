"""
Provider-agnostic LLM client.

Supports Groq (primary), OpenAI, Anthropic, Gemini, DeepSeek.
All calls are synchronous with httpx for simplicity within the 30s budget.
"""

from __future__ import annotations
import os
import json
import httpx

# ─── Configuration ────────────────────────────────────────────────────────────

# Provider: "groq", "openai", "anthropic", "gemini", "deepseek"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

# Default models per provider
_DEFAULT_MODELS = {
    "groq": "llama-3.1-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-2.0-flash",
    "deepseek": "deepseek-chat",
}

# API endpoints
_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
}

_TIMEOUT = 25  # seconds — leaves 5s buffer within the 30s budget


def _get_model() -> str:
    return LLM_MODEL or _DEFAULT_MODELS.get(LLM_PROVIDER, "llama-3.1-70b-versatile")


def complete(prompt: str, system: str | None = None, temperature: float = 0,
             max_tokens: int = 800) -> str:
    """
    Send a chat completion request and return the text response.
    Raises on failure after one attempt.
    """
    provider = LLM_PROVIDER.lower()

    if provider == "anthropic":
        return _complete_anthropic(prompt, system, temperature, max_tokens)
    elif provider == "gemini":
        return _complete_gemini(prompt, system, temperature, max_tokens)
    else:
        # OpenAI-compatible: groq, openai, deepseek
        return _complete_openai_compat(prompt, system, temperature, max_tokens)


def _complete_openai_compat(prompt: str, system: str | None,
                             temperature: float, max_tokens: int) -> str:
    """Works for Groq, OpenAI, DeepSeek — all OpenAI-compatible APIs."""
    endpoint = _ENDPOINTS.get(LLM_PROVIDER, _ENDPOINTS["groq"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": _get_model(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(endpoint, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _complete_anthropic(prompt: str, system: str | None,
                         temperature: float, max_tokens: int) -> str:
    body = {
        "model": _get_model(),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    headers = {
        "x-api-key": LLM_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]


def _complete_gemini(prompt: str, system: str | None,
                      temperature: float, max_tokens: int) -> str:
    model = _get_model()
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    body = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={LLM_API_KEY}"

    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(url, json=body, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
