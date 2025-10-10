# features/generate_target.py

import argparse

import numpy as np
import pandas as pd

from utils.project_path import PROJECT_ROOT


def calculate_atr(df, period: int = 14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def generate_target(
    df: pd.DataFrame,
    tp_multiplier=1.5,
    sl_multiplier=1.0,
    lookahead=12,
    target_col=None,
):
    df = df.copy().reset_index(drop=True)
    df.columns = df.columns.str.strip().str.lower()
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"[FEJL] Mangler kolonnen '{col}' i inputfilen! Fundet kolonner: {df.columns.tolist()}\n"
                f"Check at din CSV har korrekte OHLCV-kolonner."
            )
    if "atr_14" not in df.columns:
        df["atr_14"] = calculate_atr(df)
    df["tp"] = df["close"] + tp_multiplier * df["atr_14"]
    df["sl"] = df["close"] - sl_multiplier * df["atr_14"]
    target = []
    for idx in range(len(df)):
        take_profit = df.loc[idx, "tp"]
        stop_loss = df.loc[idx, "sl"]
        future_prices = df.loc[idx + 1 : idx + lookahead, ["high", "low"]]
        hit_tp = (future_prices["high"] >= take_profit).any()
        hit_sl = (future_prices["low"] <= stop_loss).any()
        if hit_tp and hit_sl:
            first_tp = future_prices["high"] >= take_profit
            first_sl = future_prices["low"] <= stop_loss
            first_hit = (first_tp | first_sl).idxmax()
            target_label = 1 if first_tp.loc[first_hit] else 0
        elif hit_tp:
            target_label = 1
        elif hit_sl:
            target_label = 0
        else:
            target_label = np.nan
        target.append(target_label)
    colname = target_col or f"target_tp{tp_multiplier}_sl{sl_multiplier}"
    df[colname] = target
    # Lad være med at droppe rækker her – det gøres centralt senere
    return df[[colname]]


def auto_detect_sep(filepath):
    with open(filepath, "r") as f:
        header = f.readline()
        if ";" in header and not "," in header:
            return ";"
        return ","


def main():
    parser = argparse.ArgumentParser(
        description="Generér target-kolonner til AI trading pipeline – flere TP/SL på én gang!"
    )
    parser.add_argument(
        "--input", type=str, default=str(PROJECT_ROOT / "data" / "BTCUSDT_1h.csv")
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "BTCUSDT_1h_with_target.csv"),
    )
    parser.add_argument(
        "--tp",
        type=str,
        default="1.5",
        help="Kommasepareret TP-multipler (fx '1.0,1.5,2.0')",
    )
    parser.add_argument(
        "--sl",
        type=str,
        default="1.0",
        help="Kommasepareret SL-multipler (fx '1.0,1.5')",
    )
    parser.add_argument("--lookahead", type=int, default=12)
    args = parser.parse_args()
    print(f"[INFO] Loader data fra: {args.input}")
    sep = auto_detect_sep(args.input)
    print(f"[INFO] Detekteret separator: '{sep}'")
    df_orig = pd.read_csv(args.input, sep=sep)
    print(f"[INFO] Kolonner i data: {df_orig.columns.tolist()} (før clean)")

    tp_list = [float(x) for x in args.tp.split(",")]
    sl_list = [float(x) for x in args.sl.split(",")]

    # Generér targets og saml alle kolonner i en dict
    targets_dict = {}
    for tp in tp_list:
        for sl in sl_list:
            colname = f"target_tp{tp}_sl{sl}"
            print(f"[INFO] Genererer target for TP={tp} x ATR, SL={sl} x ATR")
            series = generate_target(
                df_orig,
                tp_multiplier=tp,
                sl_multiplier=sl,
                lookahead=args.lookahead,
                target_col=colname,
            )
            # Pad/align med NaN så længden altid matcher df_orig
            series = series.reindex(range(len(df_orig)))
            targets_dict[colname] = series[colname]

    df_out = df_orig.copy().reset_index(drop=True)
    for col, series in targets_dict.items():
        df_out[col] = series.values

    # Drop kun de rækker hvor ALLE targets er NaN (dvs. ikke målbar strategi)
    target_cols = list(targets_dict.keys())
    df_out = df_out.dropna(subset=target_cols, how="all").reset_index(drop=True)

    print(f"[INFO] Første rækker efter target-generering:\n{df_out.head(3)}")
    df_out.to_csv(args.output, index=False)
    print(f"✅ Target-generering færdig. Gemt som: {args.output}")


if __name__ == "__main__":
    main()
