import pandas as pd
import pytest

import bot.engine as engine
import pipeline.core as pipeline_core
import utils.telegram_utils as telegram_utils


def test_send_live_metrics_drawdown_alert_formatting(monkeypatch):
    sent_messages = []
    sent_alerts = []

    monkeypatch.setattr(
        telegram_utils,
        "calculate_live_metrics",
        lambda *_a, **_k: {
            "profit_pct": -3.21,
            "win_rate": 42.0,
            "drawdown_pct": -12.3456,
            "num_trades": 9,
            "profit_factor": 0.7,
            "sharpe": -0.2,
        },
    )
    monkeypatch.setattr(telegram_utils, "check_drawdown_alert", lambda *_a, **_k: True)
    monkeypatch.setattr(telegram_utils, "check_winrate_alert", lambda *_a, **_k: False)
    monkeypatch.setattr(telegram_utils, "check_profit_alert", lambda *_a, **_k: False)
    monkeypatch.setattr(
        telegram_utils,
        "send_message",
        lambda text, **_k: sent_messages.append(text) or {"ok": True},
    )
    monkeypatch.setattr(
        telegram_utils,
        "send_signal_message",
        lambda text, **_k: sent_alerts.append(text) or {"ok": True},
    )

    telegram_utils.send_live_metrics(
        trades_df=pd.DataFrame(),
        balance_df=pd.DataFrame(),
        symbol="BTCUSDT",
        timeframe="1h",
        thresholds={"drawdown": -10, "winrate": 20, "profit": -10},
    )

    assert sent_messages, "Forventede statusbesked."
    assert sent_alerts, "Forventede drawdown-alert."
    assert sent_alerts[0].endswith("(-12.35%)")


def test_missing_model_artifacts_raise_in_pipeline_and_engine(monkeypatch):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="h"),
            "close": [100.0, 101.0, 100.5, 102.0],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [0.1, 0.2, 0.3, 0.4],
        }
    )

    monkeypatch.setattr(pipeline_core, "read_features_auto", lambda *_a, **_k: df.copy())
    monkeypatch.setattr(pipeline_core, "load_trained_feature_list", lambda: ["f1", "f2"])
    monkeypatch.setattr(
        pipeline_core,
        "run_backtest",
        lambda *_a, **_k: (pd.DataFrame({"profit": []}), pd.DataFrame({"balance": [100.0]})),
    )
    monkeypatch.setattr(
        pipeline_core,
        "calc_backtest_metrics",
        lambda *_a, **_k: {"profit_pct": 0.0, "drawdown_pct": 0.0},
    )
    monkeypatch.setattr(pipeline_core, "load_pytorch_model", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="DL-model|PyTorch"):
        pipeline_core.run_pipeline(
            features_path="dummy.csv",
            send_telegram=False,
            plot_graphs=False,
            save_graphs=False,
            log_to_tb=False,
        )

    monkeypatch.setattr(engine, "load_ml_model", lambda: (None, None))
    with pytest.raises(RuntimeError, match="ML-model/artefakter"):
        engine._generate_ensemble_signals_for_df(
            df=df.copy(),
            threshold=0.7,
            device_str="cpu",
            use_lstm=False,
            weights=[1.0, 1.0, 0.7],
        )
