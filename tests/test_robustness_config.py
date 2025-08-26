# -*- coding: utf-8 -*-
import pytest

from bot.utils.config_utils import validate_config
from bot.utils.errors import InvalidConfigError


def test_valid_config(sample_config):
    # Skal ikke kaste for en gyldig minimal config
    validate_config(sample_config)


def test_invalid_top_level_type():
    # Config skal være et dict
    with pytest.raises(InvalidConfigError):
        validate_config("ikke et dict")


def test_missing_strategy(sample_config):
    # Mindst én strategi er påkrævet
    bad = dict(sample_config)
    bad["strategies"] = []
    with pytest.raises(InvalidConfigError):
        validate_config(bad)


def test_bad_paths(sample_config):
    # data.paths.{raw, processed} skal være ikke-tomme strenge
    bad = dict(sample_config)
    bad["data"] = {"paths": {"raw": "", "processed": ""}}
    with pytest.raises(InvalidConfigError):
        validate_config(bad)


def test_bad_risk(sample_config):
    # trading.risk.max_position skal være mellem 0 og 1 inkl.
    bad = dict(sample_config)
    bad["trading"] = {"risk": {"max_position": 1.5}}
    with pytest.raises(InvalidConfigError):
        validate_config(bad)
