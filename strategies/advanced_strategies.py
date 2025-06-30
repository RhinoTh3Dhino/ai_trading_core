# strategies/advanced_strategies.py
import pandas as pd
import numpy as np

def ema_crossover_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enkel EMA9/EMA21 crossover strategi:
    Køb = EMA9 > EMA21, Sælg = EMA9 < EMA21
    """
    df = df.copy()
    df['signal'] = 0
    df.loc[df['ema_9'] > df['ema_21'], 'signal'] = 1
    df.loc[df['ema_9'] < df['ema_21'], 'signal'] = -1
    return df

def ema_rsi_regime_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Avanceret strategi: EMA9 > EMA21 & RSI14 > 50 & regime==bull
    Sælg: EMA9 < EMA21 eller RSI14 < 45
    """
    df = df.copy()
    df['signal'] = 0
    # Brug regime, hvis den findes – ellers antag bull (1)
    regime = df['regime'] if 'regime' in df.columns else 1
    df.loc[
        (df['ema_9'] > df['ema_21']) &
        (df['rsi_14'] > 50) &
        (regime == 1),
        'signal'
    ] = 1
    df.loc[
        (df['ema_9'] < df['ema_21']) |
        (df['rsi_14'] < 45),
        'signal'
    ] = -1
    return df

def ema_rsi_adx_strategy(df: pd.DataFrame, adx_threshold: float = 20) -> pd.DataFrame:
    """
    EMA + RSI + ADX filter (kun køb når trend er stærk).
    """
    df = df.copy()
    df['signal'] = 0
    # Hvis 'adx_14' ikke findes, brug bare EMA/RSI
    adx = df['adx_14'] if 'adx_14' in df.columns else 100
    df.loc[
        (df['ema_9'] > df['ema_21']) &
        (df['rsi_14'] > 50) &
        (adx > adx_threshold),
        'signal'
    ] = 1
    df.loc[
        (df['ema_9'] < df['ema_21']) |
        (df['rsi_14'] < 45) |
        (adx < adx_threshold),
        'signal'
    ] = -1
    return df

def rsi_mean_reversion(df: pd.DataFrame, upper: float = 70, lower: float = 30) -> pd.DataFrame:
    """
    Mean reversion: Køb hvis RSI < lower, sælg hvis RSI > upper.
    """
    df = df.copy()
    df['signal'] = 0
    df.loc[df['rsi_14'] < lower, 'signal'] = 1
    df.loc[df['rsi_14'] > upper, 'signal'] = -1
    return df

def regime_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksempel på ensemble: Brug forskellig strategi alt efter regime.
    - Bull: EMA/RSI
    - Bear: Mean reversion
    - Neutral: No trade
    """
    df = df.copy()
    df['signal'] = 0
    bull = (df.get('regime', 1) == 1)
    bear = (df.get('regime', 1) == -1)
    neutral = (df.get('regime', 0) == 0)
    # Bull: EMA/RSI
    df.loc[bull & (df['ema_9'] > df['ema_21']) & (df['rsi_14'] > 50), 'signal'] = 1
    df.loc[bull & ((df['ema_9'] < df['ema_21']) | (df['rsi_14'] < 45)), 'signal'] = -1
    # Bear: Mean reversion (køb når RSI < 30, sælg > 70)
    df.loc[bear & (df['rsi_14'] < 30), 'signal'] = 1
    df.loc[bear & (df['rsi_14'] > 70), 'signal'] = -1
    # Neutral: no trade (signal=0)
    return df

def voting_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksempel på ensemble/voting: Tre strategier stemmer, og signal = majoritet.
    """
    df = df.copy()
    votes = []
    # Tilføj flere strategier som stemmer herunder
    votes.append(ema_crossover_strategy(df)['signal'])
    votes.append(rsi_mean_reversion(df)['signal'])
    votes.append(ema_rsi_regime_strategy(df)['signal'])
    # Majoritetsafstemning
    vote_matrix = np.vstack(votes)
    df['signal'] = np.sign(np.sum(vote_matrix, axis=0))
    return df

# Tilføj evt. flere strategier efterhånden!
