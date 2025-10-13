from __future__ import annotations

import json
import os

from anthropic import Anthropic

_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY mangler i miljøet")
        _client = Anthropic(api_key=key)
    return _client


def ask_claude(
    system: str,
    user: str,
    model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> str:
    """
    Returnér ren tekst. Robust mod fejl – returnerer fejltekst som string.
    """
    try:
        client = _get_client()
        msg = client.messages.create(
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        chunks = [c.text for c in msg.content if getattr(c, "type", "") == "text"]
        return "".join(chunks).strip()
    except Exception as e:
        return f"[LLM error] {e}"
