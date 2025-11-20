import importlib
import os
from types import ModuleType


def reload_metrics(with_env: dict) -> ModuleType:
    for k, v in with_env.items():
        os.environ[k] = v
    import bot.live_connector.metrics as m

    importlib.reload(m)
    return m


def test_metrics_bootstrap_and_make_app(tmp_path):
    m = reload_metrics({"METRICS_AUTO_INIT": "1", "METRICS_BOOTSTRAP": "1"})
    app = m.make_metrics_app()
    assert app is not None
    # sanity: bootstrap har sat 0-observationer
    m.ensure_registered()
    m.inc_bars("okx", "BTCUSDT", 0)  # bør ikke fejle


def test_multiproc_gauge_mode(tmp_path):
    m = reload_metrics({"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)})
    m.ensure_registered()
    m.set_queue_depth(0)  # max-mode gauge bør virke uden exception


def test_time_feature_error_counts():
    import bot.live_connector.metrics as m

    m.ensure_registered()
    before = m.feature_errors_total.labels("ema", "BTCUSDT")._value.get()
    try:
        with m.time_feature("ema", "BTCUSDT"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    after = m.feature_errors_total.labels("ema", "BTCUSDT")._value.get()
    assert after == before + 1
