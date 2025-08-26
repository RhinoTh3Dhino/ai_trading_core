# -*- coding: utf-8 -*-
"""
Simpelt API-kald med timeout.
"""
from __future__ import annotations

import json
import socket
from typing import List, Dict
from urllib import request, error, parse

from .errors import APIError


def _guess_charset(headers) -> str:
    """Find charset fra Content-Type; fallback til utf-8."""
    ctype = headers.get("Content-Type", "")
    for part in ctype.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1].strip()
    return "utf-8"


def fetch_json(url: str, timeout: float = 5.0) -> Dict | List:
    """
    Henter JSON fra et endpoint. Kaster APIError på fejl/timeout.
    """
    parts = parse.urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise APIError(f"Ugyldigt URL-schema: {parts.scheme or 'mangler'}")

    req = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "LyraTradeBot/1.0",
        },
        method="GET",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", resp.getcode())
            if not (200 <= status < 300):
                raise APIError(f"HTTP status {status} fra {url}")

            raw = resp.read()
            charset = _guess_charset(resp.headers)
            text = raw.decode(charset, errors="replace")

            try:
                return json.loads(text)
            except json.JSONDecodeError as je:
                raise APIError(f"Ugyldig JSON: {je}") from None

    except (socket.timeout, TimeoutError):
        # Matcher din pytest, der monkeypatcher til TimeoutError i __enter__
        raise APIError("API-timeout") from None
    except error.HTTPError as e:
        # Medtag kort body-uddrag i fejlen (nyttigt i logs)
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = "<no body>"
        raise APIError(f"HTTPError {e.code}: {body[:200]}") from None
    except error.URLError as e:
        # URLError kan wrappe socket.timeout i .reason
        if isinstance(e.reason, socket.timeout):
            raise APIError("API-timeout") from None
        raise APIError(f"URLError: {e.reason}") from None
    except Exception as e:
        raise APIError(f"API-fejl: {e}") from None
