import importlib

from config import config


def test_all_features_unique_and_not_empty():
    assert len(config.ALL_FEATURES) > 0
    assert len(config.ALL_FEATURES) == len(set(config.ALL_FEATURES))


def test_coin_timeframe_defaults():
    assert isinstance(config.COINS, list)
    assert isinstance(config.TIMEFRAMES, list)
    assert all(isinstance(c, str) for c in config.COINS)


def test_reload_module():
    # Sikrer at import og reload virker uden fejl
    importlib.reload(config)
