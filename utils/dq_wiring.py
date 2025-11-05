# utils/dq_wiring.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from bot.live_connector.metrics import (
    inc_dq_violation,  # >> findes i din metrics.py efter sidste tilpasning
)
from bot.live_connector.metrics import (
    set_dq_freshness_minutes,  # >> findes i din metrics.py efter sidste tilpasning
)
from utils.data_contracts import DataContract
from utils.data_quality import validate


def dq_check_and_emit(
    df: pd.DataFrame,
    contract: DataContract,
    dataset_name: Optional[str] = None,
    minutes_since_last_update: Optional[float] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Kør kontraktvalidering og emit Prometheus-metrikker:
      - inc_dq_violation(contract.name, rule) for hver rule med issues
      - set_dq_freshness_minutes(dataset_name, minutes) hvis givet

    Returnerer: (ok, issues)
    """
    rep = validate(df, contract)

    if not rep.ok:
        for rule in rep.issues.keys():
            inc_dq_violation(contract.name, rule)  # Counter++

    if dataset_name is not None and minutes_since_last_update is not None:
        try:
            set_dq_freshness_minutes(dataset_name, float(minutes_since_last_update))
        except Exception:
            pass

    return rep.ok, rep.issues
