{
  "run_id": string,
  "model": string,
  "input_stats": {"trades": integer, "from": string, "to": string, "symbols": [string]},
  "metrics": {"win_rate": number, "avg_rr": number, "pnl_sum": number, "max_dd_est": number, "trade_frequency": number},
  "edge_score": integer,
  "risk_flags": [string],
  "anomalies": [{"type": string, "severity": "LOW"|"MEDIUM"|"HIGH", "message": string}],
  "recommendations": [{"param": string, "current": string, "proposed": string, "rationale": string}],
  "actionability": "HOLD"|"TWEAK"|"PAUSE",
  "confidence": number,
  "no_action_reason": string
}
