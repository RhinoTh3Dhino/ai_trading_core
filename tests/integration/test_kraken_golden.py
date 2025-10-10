# tests/integration/test_kraken_golden.py
from __future__ import annotations

import importlib
import sys
import time
from typing import Any, Dict, List

import pytest
from prometheus_client import REGISTRY, generate_latest


def _clear_prom_registry() -> None:
    """Ryd alle collectors fra default REGISTRY (så tests er deterministiske)."""
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)  # type: ignore[arg-type]
        except Exception:
            pass


def _reload_metrics(env: Dict[str, str | None] | None = None):
    """Reload bot.live_connector.metrics med kontrolleret miljø og rent REGISTRY."""
    import os

    if env:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # Fjern tidligere import
    for mod in list(sys.modules.keys()):
        if mod.startswith("bot.live_connector.metrics"):
            sys.modules.pop(mod, None)

    _clear_prom_registry()
    import bot.live_connector.metrics as m  # noqa

    importlib.reload(m)
    return m


def _flex_find_metric_lines(txt: str, metric: str, labels: Dict[str, str]) -> List[str]:
    """
    Find linjer for metric med alle angivne labels uanset label-rækkefølge.
    Ex: metric="feed_bars_total", labels={"venue":"kraken","symbol":"BTCUSDT"}.
    """
    out: List[str] = []
    for ln in txt.splitlines():
        if not ln.startswith(metric):
            continue
        if "{" not in ln or "}" not in ln:
            continue
        if all(f'{k}="{v}"' in ln for k, v in labels.items()):
            out.append(ln)
    return out


def _last_value(line: str) -> float:
    """Hent den numeriske værdi efter sidste mellemrum i en Prometheus scrapelinje."""
    return float(line.strip().split()[-1])


@pytest.mark.integ
def test_kraken_golden_replay(monkeypatch: pytest.MonkeyPatch):
    """
    Golden replay mod KrakenConnector:
    - Parser både array- og dict-formater for ohlc-1
    - Verificerer normaliserede bars (felter og tider)
    - Verificerer feed_bars_total tæller pr. symbol/venue
    Ingen netværk – alt er syntetisk.
    """
    # Frys tid så event_ts_ms bliver deterministisk
    fixed_now_ms = 1_760_000_000_000  # vilkårligt stabilt tidspunkt
    monkeypatch.setattr(time, "time", lambda: fixed_now_ms / 1000.0)

    # Genindlæs metrics rent og auto-init + bootstrap, uden multiproc
    _reload_metrics(
        {
            "PROMETHEUS_MULTIPROC_DIR": None,
            "METRICS_AUTO_INIT": "1",
            "METRICS_BOOTSTRAP": "1",
        }
    )

    # --- VIGTIGT: Tving reload af Kraken-modulet efter metrics-reload ---
    sys.modules.pop("bot.live_connector.venues.kraken", None)
    import bot.live_connector.venues.kraken as kraken_mod

    importlib.reload(kraken_mod)
    KrakenConnector = kraken_mod.KrakenConnector  # bind til reloaded modul

    # Symbol-map (internt -> venue)
    smap = {
        "BTCUSDT": {"kraken": "XBT/USDT"},
        "ETHUSDT": {"kraken": "ETH/USDT"},
    }

    # Syntetiske “golden” payloads (2 for BTC, 1 for ETH)
    # Array-form (BTC):
    msg_a: Dict[str, Any] = {
        "channel": "ohlc-1",
        "pair": "XBT/USDT",
        "data": [
            # [t, et, o, h, l, c, vwap, vol, count]  (t og et i sekunder)
            [
                1730572740,
                1730572800,
                "65000",
                "65100",
                "64900",
                "65050",
                "65040",
                "10.5",
                42,
            ]
        ],
    }
    # Dict-form (BTC):
    msg_b: Dict[str, Any] = {
        "event": "ohlc",
        "pair": "XBT/USDT",
        "interval": 1,
        "data": [
            {
                "time": 1730572800,
                "etime": 1730572860,
                "open": "65050",
                "high": "65120",
                "low": "65010",
                "close": "65100",
                "vwap": "65090",
                "vol": "8.0",
                "count": 30,
            }
        ],
    }
    # Array-form (ETH):
    msg_c: Dict[str, Any] = {
        "channel": "ohlc-1",
        "pair": "ETH/USDT",
        "data": [
            [
                1730572740,
                1730572800,
                "3200",
                "3210",
                "3190",
                "3205",
                "3204",
                "100.0",
                12,
            ]
        ],
    }

    connector = KrakenConnector(cfg={}, symbol_map=smap)
    out: List[Dict[str, Any]] = []
    for msg in (msg_a, msg_b, msg_c):
        evt = connector._parse_kline(msg)  # normaliseret bar
        assert evt is not None
        out.append(evt)

    # --- Assertions på event-felter ---
    assert len(out) == 3

    # Første BTC-bar (array)
    e0 = out[0]
    assert e0["venue"] == "kraken"
    assert e0["symbol"] == "BTCUSDT"
    assert e0["tf"] == "1m"
    assert e0["open_time"] == 1730572740 * 1000
    assert e0["close_time"] == 1730572800 * 1000
    assert e0["o"] == 65000.0 and e0["c"] == 65050.0 and e0["v"] == 10.5
    assert e0["is_final"] is True
    assert e0["event_ts_ms"] == fixed_now_ms

    # Anden BTC-bar (dict)
    e1 = out[1]
    assert e1["symbol"] == "BTCUSDT"
    assert e1["close_time"] == 1730572860 * 1000
    assert (
        e1["o"] == 65050.0
        and e1["h"] == 65120.0
        and e1["l"] == 65010.0
        and e1["c"] == 65100.0
    )

    # ETH-bar
    e2 = out[2]
    assert e2["symbol"] == "ETHUSDT"
    assert e2["close_time"] == 1730572800 * 1000
    assert e2["c"] == 3205.0 and e2["v"] == 100.0

    # --- Metrics: feed_bars_total skal tælle pr. symbol/venue ---
    scrap = generate_latest(REGISTRY).decode()

    btc_lines = _flex_find_metric_lines(
        scrap, "feed_bars_total", {"venue": "kraken", "symbol": "BTCUSDT"}
    )
    eth_lines = _flex_find_metric_lines(
        scrap, "feed_bars_total", {"venue": "kraken", "symbol": "ETHUSDT"}
    )

    # forvent 2 for BTC, 1 for ETH
    assert btc_lines, f"Fandt ikke feed_bars_total for BTCUSDT i scrape:\n{scrap[:800]}"
    assert eth_lines, f"Fandt ikke feed_bars_total for ETHUSDT i scrape:\n{scrap[:800]}"

    assert _last_value(btc_lines[-1]) >= 2.0
    assert _last_value(eth_lines[-1]) >= 1.0
