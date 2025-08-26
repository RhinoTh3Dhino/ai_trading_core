# -*- coding: utf-8 -*-
import socket
from urllib import request, error

import pytest

from bot.utils.api import fetch_json
from bot.utils.errors import APIError


def test_api_timeout_in_context_enter(monkeypatch):
    class Boom:
        def __enter__(self):
            # Simulerer timeout mens context manager åbnes
            raise TimeoutError("simuleret timeout")

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(url, timeout=5.0):
        return Boom()

    monkeypatch.setattr(request, "urlopen", fake_urlopen)

    with pytest.raises(APIError):
        fetch_json("http://example.com", timeout=0.1)


def test_api_timeout_via_urlerror(monkeypatch):
    def fake_urlopen(url, timeout=5.0):
        # Simulerer at urllib pakker en socket-timeout ind i URLError
        raise error.URLError(socket.timeout("simuleret"))

    monkeypatch.setattr(request, "urlopen", fake_urlopen)

    with pytest.raises(APIError):
        fetch_json("http://example.com", timeout=0.1)
