# tools/check_data_quality.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- Gør scriptet selvforsynende ift. imports (kørsel som "python tools/..") ---
ROOT = Path(__file__).resolve().parents[1]  # projektroden
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------------------------------

import pandas as pd  # noqa: E402

from utils.data_contracts import ColumnSpec, DataContract  # noqa: E402
from utils.data_quality import validate  # noqa: E402

# Eksempelkontrakter – udvid i dit repo-konfig (evt. config/yaml senere)
CONTRACTS = {
    "ohlcv_1h": DataContract(
        name="ohlcv_1h",
        required_cols={
            "timestamp": ColumnSpec("datetime"),
            "open": ColumnSpec("float", min_val=0),
            "high": ColumnSpec("float", min_val=0),
            "low": ColumnSpec("float", min_val=0),
            "close": ColumnSpec("float", min_val=0),
            "volume": ColumnSpec("float", min_val=0),
        },
        key_cols=("timestamp",),
        min_rows=100,
        max_dup_rate=0.0,
        max_null_rate=0.01,
    )
}

# Exit-koder (for CI)
EXIT_OK = 0
EXIT_DQ_FAIL = 1
EXIT_IO = 2
EXIT_CFG = 3


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"🚫 Fil findes ikke: {path}", file=sys.stderr)
        raise SystemExit(EXIT_IO)

    suf = path.suffix.lower()
    try:
        if suf in {".parquet", ".pq"}:
            # Foretræk pyarrow, men fallback til auto hvis ikke tilgængelig
            try:
                return pd.read_parquet(path, engine="pyarrow")
            except Exception:
                return pd.read_parquet(path)
        if suf == ".csv":
            return pd.read_csv(path)
    except Exception as e:
        print(f"🚫 Kunne ikke læse filen ({suf}): {e}", file=sys.stderr)
        raise SystemExit(EXIT_IO)

    print(f"🚫 Ukendt/ikke-understøttet filtype: {suf}", file=sys.stderr)
    raise SystemExit(EXIT_CFG)


def main() -> None:
    p = argparse.ArgumentParser("check_data_quality")
    p.add_argument("--dataset", required=True, help="Sti til CSV/Parquet")
    p.add_argument("--contract", required=True, choices=CONTRACTS.keys())
    p.add_argument(
        "--print-report",
        action="store_true",
        help="Print JSON-rapport til stdout",
    )
    args = p.parse_args()

    ds_path = Path(args.dataset)
    contract = CONTRACTS[args.contract]

    df = load_table(ds_path)
    rep = validate(df, contract)

    if args.print_report:
        print(json.dumps({"ok": rep.ok, "issues": rep.issues}, ensure_ascii=False, indent=2))

    # Kort tekstlig opsummering på stderr (nyttigt i CI-logs)
    if rep.ok:
        print(f"✅ DQ OK — dataset='{ds_path.name}' contract='{contract.name}'", file=sys.stderr)
        raise SystemExit(EXIT_OK)
    else:
        print(
            f"❌ DQ FAILED — dataset='{ds_path.name}' contract='{contract.name}' issues={rep.issues}",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_DQ_FAIL)


if __name__ == "__main__":
    main()
