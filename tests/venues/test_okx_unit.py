## tests/venues/test_okx_unit.py


"""
Letvægts unit-tests uden eksterne dependencies (ingen jsonschema-krav).
- Tester symbol-map funktionen
- Tester at parseren producerer de forventede keys/typer
"""

from bot.live_connector.venues.okx import (OKXConnector,
                                           parse_okx_candle_payload)


def test_symbol_map_to_internal():
    smap = {"BTCUSDT": {"okx": "BTC-USDT"}}
    c = OKXConnector(cfg={}, symbol_map=smap)
    assert c.to_internal_symbol("BTC-USDT") == "BTCUSDT"
    assert c.to_internal_symbol("UNKNOWN") == "UNKNOWN"  # fallback


def test_parse_okx_candle_payload_minimal():
    msg = {
        "arg": {"channel": "candle1m", "instId": "BTC-USDT"},
        "data": [
            ["1730572800000", "65000", "65100", "64900", "65050", "10", "0", "0", "1"]
        ],
    }
    events = parse_okx_candle_payload(msg, lambda s: "BTCUSDT")
    assert len(events) == 1
    e = events[0]
    # basisfelter
    for k in (
        "venue",
        "symbol",
        "tf",
        "open_time",
        "close_time",
        "o",
        "h",
        "l",
        "c",
        "v",
        "is_final",
        "event_ts_ms",
    ):
        assert k in e
    assert e["venue"] == "okx"
    assert e["symbol"] == "BTCUSDT"
    assert e["tf"] == "1m"
    assert isinstance(e["open_time"], int)
    assert isinstance(e["o"], float)
    assert isinstance(e["is_final"], bool)
