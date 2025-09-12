import pandas as pd, glob, os
from pathlib import Path
from datetime import datetime, timezone

def generate(outputs_dir="outputs/live", interval_ms=60_000):
    rows=[]
    for fp in glob.glob(os.path.join(outputs_dir,"*.parquet")):
        df = pd.read_parquet(fp)
        df = df.sort_values("ts")
        gaps = ((df["ts"].diff() - interval_ms) > 0).sum()
        gap_pct = 100 * gaps / max(len(df),1)
        rows.append({"file": Path(fp).name, "rows": len(df), "gaps": int(gaps), "gap_pct": gap_pct})
    rep = pd.DataFrame(rows).sort_values("gap_pct", ascending=False)
    html = rep.to_html(index=False)
    out = Path("outputs")/f"data_health_{datetime.now(timezone.utc).strftime('%Y%m%d')}.html"
    out.write_text(html, encoding="utf-8"); return out

if __name__=="__main__":
    p = generate(); print(f"Wrote report: {p}")
