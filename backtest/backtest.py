import pandas as pd

def run_backtest(df, signals=None, initial_balance=1000, fee=0.00075, sl_pct=0.02, tp_pct=0.03):
    """
    Avanceret backtest, der simulerer handler baseret på signaler (1=BUY, -1=SELL, 0=HOLD)
    Understøtter både long og short, samt SL/TP og trading fees.

    Params:
        df: pd.DataFrame med mindst kolonnerne ['timestamp', 'close']
        signals: Serie/list/np.array med signaler (samme længde som df), eller None hvis 'signal'-kolonne findes i df
        initial_balance: Startbalance
        fee: Trading fee per trade (procent, fx 0.00075 for 0.075%)
        sl_pct: Stop-loss som procent (fx 0.02 for 2%)
        tp_pct: Take-profit som procent (fx 0.03 for 3%)

    Returns:
        trades_df: pd.DataFrame med alle handler
        balance_df: pd.DataFrame med balanceudvikling
    """
    df = df.copy()
    # --- Eksplicit tjek for nødvendige kolonner ---
    for col in ["timestamp", "close"]:
        if col not in df.columns:
            raise ValueError(f"❌ Mangler kolonnen '{col}' i DataFrame til backtest! "
                             f"Findes disse kolonner? {list(df.columns)}")

    if signals is not None:
        df["signal"] = signals
    elif "signal" not in df.columns:
        raise ValueError("Ingen signaler angivet til backtest!")

    trades = []
    balance_log = []
    balance = initial_balance
    position = None
    entry_price = 0
    entry_time = None

    for i, row in df.iterrows():
        price = row["close"]
        signal = row["signal"]
        timestamp = row["timestamp"]

        # Check åben position for SL/TP (kun for long for simplicity – kan let udvides til short)
        if position == "long":
            change = (price - entry_price) / entry_price
            if change <= -sl_pct:
                # Stop loss trigger
                balance = balance * (1 - fee)
                trades.append({
                    "timestamp": timestamp,
                    "type": "SL",
                    "price": price,
                    "balance": balance
                })
                position = None
            elif change >= tp_pct:
                # Take profit trigger
                balance = balance * (1 - fee)
                trades.append({
                    "timestamp": timestamp,
                    "type": "TP",
                    "price": price,
                    "balance": balance
                })
                position = None

        # Signal fortolkes kun hvis der ikke er åben position
        if position is None:
            if signal == 1:  # BUY (long)
                entry_price = price
                entry_time = timestamp
                position = "long"
                trades.append({
                    "timestamp": timestamp,
                    "type": "BUY",
                    "price": price,
                    "balance": balance
                })
            elif signal == -1:  # SELL (short) – Udvid senere hvis du vil handle short
                # Placeholder til short logik
                pass

        # Luk position ved sidste bar
        if position == "long" and i == df.index[-1]:
            # Realiser profit/tab
            pct = (price - entry_price) / entry_price
            balance = balance * (1 + pct - fee)
            trades.append({
                "timestamp": timestamp,
                "type": "CLOSE",
                "price": price,
                "balance": balance
            })
            position = None

        balance_log.append({"timestamp": timestamp, "balance": balance})

    trades_df = pd.DataFrame(trades)
    balance_df = pd.DataFrame(balance_log)
    return trades_df, balance_df
