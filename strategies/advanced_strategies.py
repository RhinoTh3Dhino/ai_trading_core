# strategies/advanced_strategies.py

import pandas as pd
import numpy as np

# --------- Simple og avancerede trading-strategier ---------

def ema_crossover_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enkel EMA9/EMA21 crossover strategi.
    Signal: Køb hvis EMA9 > EMA21, Sælg hvis EMA9 < EMA21.
    """
    df = df.copy()
    if not all(col in df.columns for col in ["ema_9", "ema_21"]):
        df['signal'] = 0
        return df
    df['signal'] = np.where(df['ema_9'] > df['ema_21'], 1, np.where(df['ema_9'] < df['ema_21'], -1, 0))
    return df

def ema_rsi_regime_strategy(df: pd.DataFrame, rsi_entry=50, rsi_exit=45) -> pd.DataFrame:
    """
    Avanceret: Køb hvis EMA9 > EMA21 & RSI14 > rsi_entry & regime==bull.
    Sælg: EMA9 < EMA21 eller RSI14 < rsi_exit.
    """
    df = df.copy()
    for col in ["ema_9", "ema_21", "rsi_14"]:
        if col not in df.columns:
            df['signal'] = 0
            return df
    regime = df['regime'] if 'regime' in df.columns else pd.Series(1, index=df.index)
    df['signal'] = 0
    df.loc[(df['ema_9'] > df['ema_21']) & (df['rsi_14'] > rsi_entry) & (regime == 1), 'signal'] = 1
    df.loc[(df['ema_9'] < df['ema_21']) | (df['rsi_14'] < rsi_exit), 'signal'] = -1
    # Robust regime-filter: kun trade i bull
    df['signal'] = np.where(regime == 1, df['signal'], 0)
    return df

def ema_rsi_adx_strategy(df: pd.DataFrame, adx_threshold: float = 20) -> pd.DataFrame:
    """
    EMA + RSI + ADX (kun køb når trend er stærk).
    """
    df = df.copy()
    for col in ["ema_9", "ema_21", "rsi_14"]:
        if col not in df.columns:
            df['signal'] = 0
            return df
    adx = df['adx_14'] if 'adx_14' in df.columns else pd.Series(100, index=df.index)
    regime = df['regime'] if 'regime' in df.columns else pd.Series(1, index=df.index)
    df['signal'] = 0
    df.loc[
        (df['ema_9'] > df['ema_21']) &
        (df['rsi_14'] > 50) &
        (adx > adx_threshold) &
        (regime == 1),
        'signal'
    ] = 1
    df.loc[
        (df['ema_9'] < df['ema_21']) |
        (df['rsi_14'] < 45) |
        (adx < adx_threshold),
        'signal'
    ] = -1
    # Robust regime-filter
    df['signal'] = np.where(regime == 1, df['signal'], 0)
    return df

def rsi_mean_reversion(df: pd.DataFrame, upper: float = 70, lower: float = 30) -> pd.DataFrame:
    """
    Mean reversion: Køb hvis RSI < lower, sælg hvis RSI > upper.
    """
    df = df.copy()
    if "rsi_14" not in df.columns:
        df['signal'] = 0
        return df
    df['signal'] = 0
    df.loc[df['rsi_14'] < lower, 'signal'] = 1
    df.loc[df['rsi_14'] > upper, 'signal'] = -1
    return df

def regime_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensemble: forskellig strategi alt efter regime.
    Bull: EMA/RSI
    Bear: Mean reversion
    Neutral: No trade
    """
    df = df.copy()
    for col in ["ema_9", "ema_21", "rsi_14"]:
        if col not in df.columns:
            df['signal'] = 0
            return df
    regime = df.get('regime', pd.Series(1, index=df.index))
    df['signal'] = 0
    # Bull: EMA/RSI
    mask_bull = (regime == 1) & (df['ema_9'] > df['ema_21']) & (df['rsi_14'] > 50)
    mask_bull_exit = (regime == 1) & ((df['ema_9'] < df['ema_21']) | (df['rsi_14'] < 45))
    df.loc[mask_bull, 'signal'] = 1
    df.loc[mask_bull_exit, 'signal'] = -1
    # Bear: Mean reversion
    mask_bear = regime == -1
    df.loc[mask_bear & (df['rsi_14'] < 30), 'signal'] = 1
    df.loc[mask_bear & (df['rsi_14'] > 70), 'signal'] = -1
    # Neutral: signal=0
    return df

def voting_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensemble/voting: Flere strategier stemmer, signal = majoritet.
    Kun trade i bull-regime, ellers hold (signal=0).
    """
    df = df.copy()
    strategies = [
        ema_crossover_strategy,
        rsi_mean_reversion,
        ema_rsi_regime_strategy,
        # ema_rsi_adx_strategy,   # Kan nemt tilføjes for endnu mere robust voting!
    ]
    votes = [func(df)['signal'].fillna(0).astype(int).values for func in strategies]
    vote_matrix = np.vstack(votes)
    # Majoritetsafstemning
    df['signal'] = np.sign(np.sum(vote_matrix, axis=0))
    # Kun trade i bull-regime (ellers signal=0)
    if 'regime' in df.columns:
        df['signal'] = np.where(df['regime'] == 1, df['signal'], 0)
    return df

def combined_supertrend_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kombiner med Supertrend (kun hvis feature findes)
    """
    df = df.copy()
    if "supertrend_10_3" not in df.columns:
        df['signal'] = 0
        return df
    df['signal'] = df['supertrend_10_3']
    return df

# --- Bonus: Adaptive SL/TP forslag (kan flyttes til paper_trader hvis ønsket) ---
def add_adaptive_sl_tp(df: pd.DataFrame, atr_col: str = "atr_14", atr_mult_sl=1.5, atr_mult_tp=3.0):
    """
    Tilføjer adaptive SL/TP kolonner baseret på ATR.
    """
    df = df.copy()
    if atr_col in df.columns:
        df['sl_pct'] = df[atr_col] * atr_mult_sl / df['close']
        df['tp_pct'] = df[atr_col] * atr_mult_tp / df['close']
    else:
        df['sl_pct'] = 0.02
        df['tp_pct'] = 0.04
    return df
