import ccxt, time
from typing import List
from data.schemas import Bar

def rest_catchup(symbol: str, venue: str, interval: str, since_ms: int, limit: int=200) -> List[Bar]:
    ex = getattr(ccxt, venue)()
    ex.enableRateLimit = True
    ohlcv = ex.fetch_ohlcv(symbol.replace("USDT","/USDT"), timeframe=interval, since=since_ms, limit=limit)
    bars = []
    for t,o,h,l,c,v in ohlcv:
        bars.append(Bar(ts=int(t), symbol=symbol, venue=venue, interval=interval,
                        open=float(o), high=float(h), low=float(l), close=float(c), volume=float(v), is_final=True))
    return bars
