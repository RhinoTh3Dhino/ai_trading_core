import json, pathlib
from bot.live_connector.venues.okx import OKXConnector

def test_okx_golden_replay():
    cfg = {"ws":{"subs":[{"channel":"candle1m","instId":"BTC-USDT"}]}}
    sm  = {"BTCUSDT":{"okx":"BTC-USDT"}}
    out = []
    c = OKXConnector(cfg, sm, out.append, ws_client=None)

    p = pathlib.Path("tests/golden/okx_btcusdt_1m.jsonl")
    frames = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else [
        {"arg":{"channel":"candle1m","instId":"BTC-USDT"},"data":[["1725553200000","61000","61100","60900","61050","12.3","123.4"]]},
        {"arg":{"channel":"candle1m","instId":"BTC-USDT"},"data":[["1725553260000","61050","61120","61000","61080","10.0","90.0"]]}
    ]
    for m in frames:
        k = c._parse_kline(m)
        assert k is not None
        out.append(k)

    assert all(b["venue"]=="okx" and b["symbol"] in ("BTCUSDT","ETHUSDT") for b in out)
    assert len(out) == len(frames)
