import asyncio
import json

import yaml
from prometheus_client import start_http_server

from bot.live_connector.venues.okx import OKXConnector
from bot.live_connector.ws_client import WSClient


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def on_kline(evt: dict):
    # TODO: send til event-bus/pipeline. Midlertidigt: skriv til log/STDOUT
    print(json.dumps(evt, separators=(",", ":")))


async def main():
    cfg = load_yaml("config/venue_okx.yaml")
    smap = load_yaml("config/symbol_map.yaml")
    ws = WSClient()
    c = OKXConnector(cfg, smap, on_kline, ws)
    await c.run()


if __name__ == "__main__":
    start_http_server(9000)  # /metrics
    asyncio.run(main())
