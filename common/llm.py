"""Thin OpenAI wrapper shared by both parts: one place for the model name, the
API key lookup (env or Streamlit secrets), and schema-constrained calls."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


def secret(name: str, default: str | None = None) -> str | None:
    """Env var first, then Streamlit secrets (for the hosted demo)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:  # noqa: BLE001 -- no secrets file / not running under Streamlit
        return default


def chat_model() -> str:
    return secret("OPENAI_CHAT_MODEL", "gpt-5.6")


@lru_cache(maxsize=1)
def client():
    from openai import OpenAI

    key = secret("OPENAI_API_KEY")
    if key:
        os.environ.setdefault("OPENAI_API_KEY", key)  # the pipeline's embedder reads the env
    return OpenAI(api_key=key)


def explain_llm_error(exc: BaseException) -> str | None:
    """A message a user can act on for an OpenAI failure, or None if ``exc`` isn't one."""
    import openai

    if isinstance(exc, openai.RateLimitError):
        if {"insufficient_quota", "credit_balance_exhausted"} & {getattr(exc, "type", None), getattr(exc, "code", None)}:
            return ("The OpenAI account has no credits left. Add credits at "
                    "https://platform.openai.com/settings/organization/billing, then try again.")
        return "OpenAI is rate-limiting requests right now. Wait a minute and try again."
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in .env or the Streamlit secrets."
    if isinstance(exc, openai.PermissionDeniedError):
        return f"This OpenAI key isn't allowed to use the requested model ({chat_model()}) or project."
    if isinstance(exc, openai.NotFoundError):
        return f"OpenAI doesn't recognise the model. Check OPENAI_CHAT_MODEL (currently {chat_model()})."
    if isinstance(exc, openai.APIConnectionError):   # includes timeouts
        return "Couldn't reach OpenAI. Check the internet connection and try again."
    if isinstance(exc, openai.InternalServerError):
        return "OpenAI had a server error. Try again in a moment."
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI rejected the request ({exc.status_code}): {exc.message}"
    if isinstance(exc, openai.OpenAIError):          # e.g. no API key configured at all
        return f"OpenAI isn't configured: {exc}. Set OPENAI_API_KEY in .env or the Streamlit secrets."
    return None


def structured(system: str, user: str, schema: type[T], *, model: str | None = None) -> T:
    """One schema-constrained completion. The response is guaranteed to parse into
    ``schema`` (OpenAI structured outputs), so downstream code never string-parses."""
    response = client().chat.completions.parse(
        model=model or chat_model(),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=schema,
    )
    message = response.choices[0].message
    if message.parsed is None:
        raise RuntimeError(f"Model refused or returned no structured output: {message.refusal!r}")
    return message.parsed
