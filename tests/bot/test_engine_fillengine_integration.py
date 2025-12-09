# tests/bot/test_engine_fillengine_integration.py
import numpy as np
import pandas as pd

from bot import engine


def _make_dummy_df(n: int = 10) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="H")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": np.linspace(100.0, 110.0, n),
            "volume": np.linspace(1.0, 2.0, n),
        }
    )


def test_run_bt_with_rescue_kalder_fillenginev2(monkeypatch):
    calls = []

    class DummyFillEngine:
        def __init__(self, cfg):
            # vi vil bare bekræfte at vi bliver instansieret
            self.cfg = cfg

        def simulate_order(self, order, snapshot):
            calls.append((order, snapshot))

            class Res:
                # minimal fill-respons, der ligner den rigtige
                def __init__(self):
                    self.fills = []

                    # "status" skal findes, men betyder ikke noget her
                    self.status = "rejected"

            return Res()

    # Monkeypatch engine.FillEngineV2 til vores dummy
    monkeypatch.setattr(engine, "FillEngineV2", DummyFillEngine)

    df = _make_dummy_df(10)
    signals = np.array([1] * 5 + [0] * 5)

    trades, balance = engine._run_bt_with_rescue(df, signals)

    # Vi forventer at FillEngineV2 blev forsøgt brugt mindst én gang
    assert calls, "Forventer at FillEngineV2.simulate_order blev kaldt mindst én gang"

    # Og at vi får et gyldigt fallback-backtest output (rescue)
    assert trades is not None
    assert balance is not None
    assert "balance" in balance.columns or "equity" in balance.columns
    assert len(balance) > 0
