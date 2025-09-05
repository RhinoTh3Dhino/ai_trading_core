from pathlib import Path
import pandas as pd
from typing import Dict, Any

def load_recent_trades(root: Path, n: int = 20) -> pd.DataFrame:
    files = sorted((root/"outputs"/"logs").glob("*trades*.csv"))
    if not files:
        raise FileNotFoundError("No trades CSV found under outputs/logs/")
    df = pd.read_csv(files[-1])
    return df.tail(n).copy()

def summarize_for_prompt(df: pd.DataFrame) -> Dict[str, Any]:
    syms = sorted(map(str, set(df["SYMBOL"])))[:10] if "SYMBOL" in df.columns else []
    return {
        "trades": int(len(df)),
        "symbols": syms,
        "from": str(df["ENTRY"].min())[:19] if "ENTRY" in df.columns else "",
        "to": str(df["EXIT"].max())[:19] if "EXIT" in df.columns else "",
    }
