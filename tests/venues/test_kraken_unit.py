# tests/venues/test_kraken_unit.py
import time

from bot.live_connector.venues.kraken import (KrakenConnector,
                                              parse_kraken_candle_payload)


def test_symbol_map_to_internal():
    smap = {"BTCUSDT": {"kraken": "XBT/USDT"}}
    c = KrakenConnector(cfg={}, symbol_map=smap)
    assert c.to_internal_symbol("XBT/USDT") == "BTCUSDT"
    assert c.to_internal_symbol("XBTUSDT") == "BTCUSDT"  # slash-insensitive


def test_parse_kraken_payload_array():
    smap = {"BTCUSDT": {"kraken": "XBT/USDT"}}
    msg = {
        "channel": "ohlc-1",
        "pair": "XBT/USDT",
        "data": [
            [
                1730572740,
                1730572800,
                "65000",
                "65100",
                "64900",
                "65050",
                "65040",
                "10",
                42,
            ]
        ],
    }
    evt = parse_kraken_candle_payload(msg, smap)
    assert evt and evt["venue"] == "kraken"
    assert evt["symbol"] == "BTCUSDT"
    assert evt["tf"] == "1m"
    assert evt["close_time"] == 1730572800 * 1000
    assert evt["c"] == 65050.0
    assert evt["v"] == 10.0
    assert evt["is_final"] is True
    assert evt["event_ts_ms"] <= int(time.time() * 1000)


def test_parse_kraken_payload_dict():
    smap = {"BTCUSDT": {"kraken": "XBT/USDT"}}
    msg = {
        "event": "ohlc",
        "pair": "XBT/USDT",
        "interval": 1,
        "data": [
            {
                "time": 1730572740,
                "etime": 1730572800,
                "open": "65000",
                "high": "65100",
                "low": "64900",
                "close": "65050",
                "vwap": "65040",
                "vol": "10",
                "count": 42,
            }
        ],
    }
    evt = parse_kraken_candle_payload(msg, smap)
    assert evt and evt["symbol"] == "BTCUSDT" and evt["c"] == 65050.0
