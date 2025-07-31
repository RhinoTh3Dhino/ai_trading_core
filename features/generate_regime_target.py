from utils.project_path import PROJECT_ROOT
import pandas as pd
import numpy as np
import argparse


def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def get_regime(df):
    # Trend-regime: ema_9 > ema_21, ellers range
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["regime"] = (df["ema_9"] > df["ema_21"]).astype(int)
    return df


def generate_regime_targets(df, lookahead=12):
    # ATR-beregning
    if "atr_14" not in df.columns:
        df["atr_14"] = calculate_atr(df)
    # TP/SL for hvert regime
    df["tp_trend"] = df["close"] + 2.0 * df["atr_14"]
    df["sl_trend"] = df["close"] - 1.0 * df["atr_14"]
    df["tp_range"] = df["close"] + 0.7 * df["atr_14"]
    df["sl_range"] = df["close"] - 0.7 * df["atr_14"]

    targets = []
    for idx in range(len(df)):
        regime = df.loc[idx, "regime"]
        close = df.loc[idx, "close"]
        if regime == 1:  # Trend
            tp = df.loc[idx, "tp_trend"]
            sl = df.loc[idx, "sl_trend"]
        else:  # Range
            tp = df.loc[idx, "tp_range"]
            sl = df.loc[idx, "sl_range"]
        # Fremtidige priser
        future = df.loc[idx + 1 : idx + lookahead, ["high", "low"]]
        hit_tp = (future["high"] >= tp).any()
        hit_sl = (future["low"] <= sl).any()
        # Label: 1 = TP først, 0 = SL først, nan = ingen hit
        if hit_tp and hit_sl:
            first_tp = future["high"] >= tp
            first_sl = future["low"] <= sl
            first_hit = (first_tp | first_sl).idxmax()
            label = 1 if first_tp.loc[first_hit] else 0
        elif hit_tp:
            label = 1
        elif hit_sl:
            label = 0
        else:
            label = np.nan
        targets.append(label)
    df["target_regime_adapt"] = targets
    df.dropna(subset=["target_regime_adapt"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def auto_detect_sep(filepath):
    with open(filepath, "r") as f:
        header = f.readline()
        if ";" in header and not "," in header:
            return ";"
        return ","


def main():
    parser = argparse.ArgumentParser(
        description="Regime-adaptiv target-generator (trend/range)"
    )
    parser.add_argument(
        "--input", type=str, default=str(PROJECT_ROOT / "data" / "BTCUSDT_1h.csv")
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "BTCUSDT_1h_with_regime_target.csv"),
    )
    parser.add_argument("--lookahead", type=int, default=12)
    args = parser.parse_args()

    sep = auto_detect_sep(args.input)
    df = pd.read_csv(args.input, sep=sep)
    print("[INFO] Oprindelige kolonner:", df.columns.tolist())
    df = get_regime(df)
    df = generate_regime_targets(df, lookahead=args.lookahead)
    print(
        "[INFO] Regime target-fordeling:\n",
        df["target_regime_adapt"].value_counts(dropna=False),
    )
    df.to_csv(args.output, index=False)
    print(f"✅ Gemte regime-adaptive targets i: {args.output}")


if __name__ == "__main__":
    main()
