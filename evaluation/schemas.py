# evaluation/schemas.py
from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict  # Pydantic v2-kompatibel

class Anomaly(BaseModel):
    type: str
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    message: str

class Recommendation(BaseModel):
    param: str
    current: str
    proposed: str
    rationale: str

class Metrics(BaseModel):
    win_rate: float
    avg_rr: float
    pnl_sum: float
    max_dd_est: float
    trade_frequency: float

class InputStats(BaseModel):
    trades: int
    from_: str = Field(alias="from")
    to: str
    symbols: List[str]

class EdgeEvalV1(BaseModel):
    run_id: str
    model: str
    input_stats: InputStats
    metrics: Metrics
    edge_score: int
    risk_flags: List[str] = Field(default_factory=list, max_length=10)
    anomalies: List[Anomaly] = Field(default_factory=list, max_length=10)
    recommendations: List[Recommendation] = Field(default_factory=list, max_length=5)  # Ændret til Field
    actionability: Literal["HOLD", "TWEAK", "PAUSE"]
    confidence: float
    no_action_reason: str

    class Config:
        # Pydantic v2 konfiguration
        model_config = ConfigDict(populate_by_name=True, from_attributes=True)

