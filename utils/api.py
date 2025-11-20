# -*- coding: utf-8 -*-
"""
Simpelt, robust API-lag til JSON-kald med timeout og defensiv fejlhåndtering.

Bevarer:
- fetch_json(url, timeout)  → bruges i tests og andre moduler.

Tilføjer:
- request_json(method, url, params=None, data=None, headers=None, timeout=5.0, max_bytes=2_000_000)
  Fleksibel helper, der dækker GET/POST m.m. og bruges af fetch_json.

Fejlhåndtering:
- Giver APIError med korte, nyttige fejlbeskeder (inkl. kort body ved HTTPError).
- Fanger både socket.timeout, TimeoutError og URLError med .reason=timeout (matcher dine tests).
- Forsøger at dekode body ud fra charset i Content-Type, falder tilbage til utf-8.

Defensivt:
- Begrænser maks. body-størrelse (default 2MB) for at undgå OOM/problemer.
- Accepterer JSON og problem+json.
- Afviser ugyldige URL-schemes.
"""
from __future__ import annotations

import io
import json
import socket
from typing import Any, Dict, List, Mapping, Optional
from urllib import error, parse, request

from .errors import APIError

__all__ = ["fetch_json", "request_json"]


def _guess_charset(headers: Mapping[str, str]) -> str:
    """Find charset fra Content-Type; fallback til utf-8."""
    ctype = headers.get("Content-Type", "")
    for part in str(ctype).split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1].strip() or "utf-8"
    return "utf-8"


def _build_url_with_params(url: str, params: Optional[Mapping[str, Any]]) -> str:
    if not params:
        return url
    parts = parse.urlparse(url)
    query = dict(parse.parse_qsl(parts.query, keep_blank_values=True))
    # opdater/tilføj parametre
    for k, v in params.items():
        query[str(k)] = "" if v is None else str(v)
    new_qs = parse.urlencode(query)
    new_parts = parts._replace(query=new_qs)
    return parse.urlunparse(new_parts)


def _read_limited(resp, max_bytes: int) -> bytes:
    """
    Læs op til max_bytes bytes fra response-stream defensivt.
    """
    buf = io.BytesIO()
    chunk = True
    to_read = max_bytes
    while chunk and to_read > 0:
        chunk = resp.read(min(65536, to_read))
        if not chunk:
            break
        buf.write(chunk)
        to_read -= len(chunk)
    if to_read == 0:  # vi har ramt grænsen – fortsæt ikke
        # Tøm resten (uden at gemme) for at lukke forbindelsen pænt
        try:
            while resp.read(65536):
                pass
        except Exception:
            pass
    return buf.getvalue()


def request_json(
    method: str,
    url: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    data: Optional[Mapping[str, Any] | bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 5.0,
    max_bytes: int = 2_000_000,
) -> Dict | List:
    """
    Generisk JSON-request.

    Args:
        method: "GET" | "POST" | ...
        url: basis-URL.
        params: dict med query-params der merges ind i url.
        data: dict (JSON-encodes) eller bytes (sendes som-is).
        headers: ekstra headers. Accept/UA udfyldes fornuftigt som default.
        timeout: socket-timeout i sekunder.
        max_bytes: maks bytes der læses fra svaret.

    Returns:
        Dict eller List (parsed JSON).

    Raises:
        APIError på netværksfejl, timeouts, HTTP-fejl og JSON-decode-fejl.
    """
    if not isinstance(url, str) or not url:
        raise APIError("Ugyldigt URL: tom streng")
    parts = parse.urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise APIError(f"Ugyldigt URL-schema: {parts.scheme or 'mangler'}")

    url_final = _build_url_with_params(url, params)

    # Default headers
    base_headers = {
        "Accept": "application/json, application/problem+json",
        "User-Agent": "LyraTradeBot/1.0",
    }
    if headers:
        base_headers.update(headers)

    # Body
    body_bytes: Optional[bytes] = None
    if data is not None:
        if isinstance(data, (bytes, bytearray)):
            body_bytes = bytes(data)
        else:
            try:
                body_bytes = json.dumps(data).encode("utf-8")
            except Exception as je:
                raise APIError(f"Kunne ikke JSON-serialisere body: {je}") from None
        # Sæt Content-Type hvis ikke allerede angivet
        if not any(h.lower() == "content-type" for h in base_headers.keys()):
            base_headers["Content-Type"] = "application/json"

    req = request.Request(
        url_final,
        headers=base_headers,
        method=method.upper(),
        data=body_bytes,
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            if status is None:
                # Py3.9 kompatibilitet
                status = resp.getcode()

            if not (200 <= int(status) < 300):
                raise APIError(f"HTTP status {status} fra {url_final}")

            raw = _read_limited(resp, max_bytes=max_bytes)
            charset = _guess_charset(getattr(resp, "headers", {}))
            text = raw.decode(charset, errors="replace")

            try:
                return json.loads(text)
            except json.JSONDecodeError as je:
                raise APIError(f"Ugyldig JSON: {je}") from None

    except (socket.timeout, TimeoutError):
        # Matcher pytest, der kan rejse TimeoutError i __enter__
        raise APIError("API-timeout") from None
    except error.HTTPError as e:
        # Medtag kort body-uddrag i fejlen (nyttigt i logs)
        try:
            body = e.read()  # type: ignore[attr-defined]
            # prøv at dekode – vi kender ikke charset i dette branch
            body_txt = (
                body.decode("utf-8", errors="replace")
                if isinstance(body, (bytes, bytearray))
                else str(body)
            )
        except Exception:
            body_txt = "<no body>"
        raise APIError(f"HTTPError {e.code}: {body_txt[:200]}") from None
    except error.URLError as e:
        # URLError kan wrappe socket.timeout i .reason
        if isinstance(getattr(e, "reason", None), socket.timeout):
            raise APIError("API-timeout") from None
        raise APIError(f"URLError: {getattr(e, 'reason', e)}") from None
    except Exception as e:
        raise APIError(f"API-fejl: {e}") from None


def fetch_json(url: str, timeout: float = 5.0) -> Dict | List:
    """
    Simpelt GET-kald der henter JSON fra et endpoint.

    Bevidst tynd wrapper om request_json for at bevare eksisterende API.
    """
    return request_json("GET", url, timeout=timeout)
