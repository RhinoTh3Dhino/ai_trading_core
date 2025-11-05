# utils/data_contracts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence

DType = Literal["float", "int", "str", "datetime", "bool"]


@dataclass(frozen=True)
class ColumnSpec:
    dtype: DType
    allow_null: bool = False
    min_val: Optional[float] = None
    max_val: Optional[float] = None


@dataclass(frozen=True)
class DataContract:
    name: str
    required_cols: Dict[str, ColumnSpec]
    key_cols: Sequence[str] = ()
    min_rows: int = 1
    max_dup_rate: float = 0.02  # 2 %
    max_null_rate: float = 0.01  # 1 %
