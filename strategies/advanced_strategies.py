# strategies/advanced_strategies.py

import numpy as np
import pandas as pd

# --------- Simple og avancerede trading-strategier ---------


def ema_crossover_strategy(df: pd.DataFrame, fast=9, slow=21) -> pd.DataFrame:
    """
    Enkel EMA crossover strategi (default: EMA9/EMA21).
    Signal: Køb hvis EMA_fast > EMA_slow, Sælg hvis EMA_fast < EMA_slow.
    """
    df = df.copy()
    fast_col = f"ema_{fast}"
    slow_col = f"ema_{slow}"
    # Support både klassisk og default (ema_9/ema_21)
    if not all(col in df.columns for col in [fast_col, slow_col]):
        if all(col in df.columns for col in ["ema_9", "ema_21"]):
            fast_col, slow_col = "ema_9", "ema_21"
        else:
            df["signal"] = 0
            return df
    df["signal"] = np.where(
        df[fast_col] > df[slow_col], 1, np.where(df[fast_col] < df[slow_col], -1, 0)
    )
    return df


def ema_rsi_regime_strategy(
    df: pd.DataFrame, rsi_entry=50, rsi_exit=45, regime_col="regime", regime_val="bull"
) -> pd.DataFrame:
    """
    Avanceret: Køb hvis EMA9 > EMA21 & RSI14 > rsi_entry & regime==bull.
    Sælg: EMA9 < EMA21 eller RSI14 < rsi_exit.
    Kun handel i bull regime.
    """
    df = df.copy()
    for col in ["ema_9", "ema_21", "rsi_14"]:
        if col not in df.columns:
            df["signal"] = 0
            return df
    # Support for regime som både tekst og int
    regime = df[regime_col] if regime_col in df.columns else pd.Series("bull", index=df.index)
    is_bull = (regime == regime_val) | (regime == 1)
    df["signal"] = 0
    buy_mask = (df["ema_9"] > df["ema_21"]) & (df["rsi_14"] > rsi_entry) & is_bull
    sell_mask = (df["ema_9"] < df["ema_21"]) | (df["rsi_14"] < rsi_exit)
    df.loc[buy_mask, "signal"] = 1
    df.loc[sell_mask, "signal"] = -1
    df["signal"] = np.where(is_bull, df["signal"], 0)
    return df


def ema_rsi_adx_strategy(
    df: pd.DataFrame, adx_threshold=20, regime_col="regime", regime_val="bull"
) -> pd.DataFrame:
    """
    EMA + RSI + ADX filter (kun køb når trend er stærk og regime bull).
    """
    df = df.copy()
    for col in ["ema_9", "ema_21", "rsi_14"]:
        if col not in df.columns:
            df["signal"] = 0
            return df
    adx = df["adx_14"] if "adx_14" in df.columns else pd.Series(100, index=df.index)
    regime = df[regime_col] if regime_col in df.columns else pd.Series("bull", index=df.index)
    is_bull = (regime == regime_val) | (regime == 1)
    df["signal"] = 0
    buy_mask = (df["ema_9"] > df["ema_21"]) & (df["rsi_14"] > 50) & (adx > adx_threshold) & is_bull
    sell_mask = (df["ema_9"] < df["ema_21"]) | (df["rsi_14"] < 45) | (adx < adx_threshold)
    df.loc[buy_mask, "signal"] = 1
    df.loc[sell_mask, "signal"] = -1
    df["signal"] = np.where(is_bull, df["signal"], 0)
    return df


def rsi_mean_reversion(df: pd.DataFrame, upper=70, lower=30) -> pd.DataFrame:
    """
    Mean reversion: Køb hvis RSI < lower, sælg hvis RSI > upper.
    """
    df = df.copy()
    if "rsi_14" not in df.columns:
        df["signal"] = 0
        return df
    df["signal"] = 0
    df.loc[df["rsi_14"] < lower, "signal"] = 1
    df.loc[df["rsi_14"] > upper, "signal"] = -1
    return df


def regime_ensemble(df: pd.DataFrame, regime_col="regime") -> pd.DataFrame:
    """
    Ensemble: forskellig strategi alt efter regime.
    Bull: EMA/RSI strategi, Bear: Mean reversion, Neutral: Ingen handel.
    """
    df = df.copy()
    for col in ["ema_9", "ema_21", "rsi_14"]:
        if col not in df.columns:
            df["signal"] = 0
            return df
    regime = df.get(regime_col, pd.Series("bull", index=df.index))
    df["signal"] = 0
    # Bull regime
    mask_bull = (
        ((regime == "bull") | (regime == 1)) & (df["ema_9"] > df["ema_21"]) & (df["rsi_14"] > 50)
    )
    mask_bull_exit = ((regime == "bull") | (regime == 1)) & (
        (df["ema_9"] < df["ema_21"]) | (df["rsi_14"] < 45)
    )
    df.loc[mask_bull, "signal"] = 1
    df.loc[mask_bull_exit, "signal"] = -1
    # Bear regime
    mask_bear = (regime == "bear") | (regime == -1)
    df.loc[mask_bear & (df["rsi_14"] < 30), "signal"] = 1
    df.loc[mask_bear & (df["rsi_14"] > 70), "signal"] = -1
    # Neutral regime = 0 signal (default)
    return df


def voting_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensemble/voting: Flere strategier stemmer, signal = majoritet.
    Kun handel i bull regime, ellers hold position (signal=0).
    """
    df = df.copy()
    strategies = [
        ema_crossover_strategy,
        rsi_mean_reversion,
        ema_rsi_regime_strategy,
        # ema_rsi_adx_strategy,  # Kan aktiveres
    ]
    votes = [func(df)["signal"].fillna(0).astype(int).values for func in strategies]
    vote_matrix = np.vstack(votes)
    majority = np.sign(np.sum(vote_matrix, axis=0))
    if "regime" in df.columns:
        is_bull = (df["regime"] == "bull") | (df["regime"] == 1)
        df["signal"] = np.where(is_bull, majority, 0)
    else:
        df["signal"] = majority
    return df


def combined_supertrend_strategy(df: pd.DataFrame, st_col="supertrend_10_3") -> pd.DataFrame:
    """
    Kombiner med Supertrend (kun hvis feature findes).
    """
    df = df.copy()
    if st_col not in df.columns:
        df["signal"] = 0
        return df
    df["signal"] = df[st_col]
    return df


# --- Adaptive SL/TP baseret på ATR ---
def add_adaptive_sl_tp(
    df: pd.DataFrame, atr_col="atr_14", atr_mult_sl=1.5, atr_mult_tp=3.0
) -> pd.DataFrame:
    """
    Tilføjer adaptive stop loss (SL) og take profit (TP) niveauer baseret på ATR.
    """
    df = df.copy()
    if atr_col in df.columns and "close" in df.columns:
        df["sl_pct"] = df[atr_col] * atr_mult_sl / df["close"]
        df["tp_pct"] = df[atr_col] * atr_mult_tp / df["close"]
    else:
        df["sl_pct"] = 0.02
        df["tp_pct"] = 0.04
    return df
