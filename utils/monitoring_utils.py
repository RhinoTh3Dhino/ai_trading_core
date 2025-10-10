# utils/monitoring_utils.py

import numpy as np

# === Hent alle monitoring thresholds og regler fra config ===
try:
    from config.monitoring_config import (ALARM_THRESHOLDS, ALERT_ON_DRAWNDOWN,
                                          ALERT_ON_PROFIT, ALERT_ON_WINRATE,
                                          ENABLE_MONITORING)
except ImportError:
    ALARM_THRESHOLDS = {"drawdown": -20, "winrate": 20, "profit": -10}
    ALERT_ON_DRAWNDOWN = True
    ALERT_ON_WINRATE = True
    ALERT_ON_PROFIT = True
    ENABLE_MONITORING = True


def calculate_live_metrics(trades_df, balance_df, initial_balance=1000):
    """
    Udregn de vigtigste nøgletal for live-simulering:
    - profit_pct
    - win_rate
    - drawdown_pct (maks)
    - antal handler
    - profit factor
    - sharpe (meget grov daglig udgave)
    """
    if trades_df is None or trades_df.empty:
        return {
            "profit_pct": 0.0,
            "win_rate": 0.0,
            "drawdown_pct": 0.0,
            "num_trades": 0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
        }

    # Profit (pct)
    final_balance = (
        balance_df["balance"].iloc[-1] if not balance_df.empty else initial_balance
    )
    profit_pct = (final_balance - initial_balance) / initial_balance * 100

    # Win-rate (% af lukkede trades med positiv profit)
    closed = trades_df[trades_df["type"].isin(["TP", "SL", "CLOSE"])]
    wins = closed[closed["profit"] > 0]
    win_rate = 100 * len(wins) / len(closed) if len(closed) > 0 else 0.0

    # Max drawdown (pct)
    if not balance_df.empty:
        cummax = balance_df["balance"].cummax()
        dd = (balance_df["balance"] - cummax) / cummax
        drawdown_pct = dd.min() * 100
    else:
        drawdown_pct = 0.0

    # Profit factor
    gross_profit = closed[closed["profit"] > 0]["profit"].sum()
    gross_loss = -closed[closed["profit"] < 0]["profit"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Simpel “sharpe ratio” på daglige returns
    returns = (
        balance_df["balance"].pct_change().dropna() if not balance_df.empty else []
    )
    sharpe = (
        (returns.mean() / returns.std()) * np.sqrt(252)
        if len(returns) > 2 and returns.std() > 0
        else 0.0
    )

    return {
        "profit_pct": round(profit_pct, 2),
        "win_rate": round(win_rate, 1),
        "drawdown_pct": round(drawdown_pct, 2),
        "num_trades": int(len(closed)),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2),
    }


def check_drawdown_alert(metrics, threshold=None):
    """
    Returnerer True hvis drawdown pct. er UNDER threshold.
    (fx threshold=-20 => alarm hvis drawdown < -20%)
    """
    if threshold is None:
        threshold = ALARM_THRESHOLDS.get("drawdown", -20)
    dd = metrics.get("drawdown_pct", 0.0)
    return dd <= threshold


def check_winrate_alert(metrics, threshold=None):
    """
    Returnerer True hvis win-rate (%) er UNDER threshold.
    (fx threshold=20 => alarm hvis win-rate < 20%)
    """
    if threshold is None:
        threshold = ALARM_THRESHOLDS.get("winrate", 20)
    wr = metrics.get("win_rate", 0.0)
    return wr < threshold


def check_profit_alert(metrics, threshold=None):
    """
    Returnerer True hvis profit (%) er UNDER threshold.
    (fx threshold=-10 => alarm hvis profit < -10%)
    """
    if threshold is None:
        threshold = ALARM_THRESHOLDS.get("profit", -10)
    profit = metrics.get("profit_pct", 0.0)
    return profit <= threshold


def send_live_metrics(
    trades_df,
    balance_df,
    symbol="",
    timeframe="",
    thresholds=None,
    alert_on_drawdown=None,
    alert_on_winrate=None,
    alert_on_profit=None,
    chat_id=None,
):
    """
    Udregner og sender live metrics samt evt. alarm-beskeder til Telegram.
    Parametre kan overrides ellers bruges globale fra config.
    """
    # Importér kun send_message lokalt for at undgå circular import!
    from utils.telegram_utils import send_message

    # Brug thresholds og flags fra config hvis ikke angivet
    thresholds = thresholds or ALARM_THRESHOLDS
    alert_on_drawdown = (
        ALERT_ON_DRAWNDOWN if alert_on_drawdown is None else alert_on_drawdown
    )
    alert_on_winrate = (
        ALERT_ON_WINRATE if alert_on_winrate is None else alert_on_winrate
    )
    alert_on_profit = ALERT_ON_PROFIT if alert_on_profit is None else alert_on_profit

    metrics = calculate_live_metrics(trades_df, balance_df)
    txt = (
        f"<b>📈 Live monitoring ({symbol} {timeframe})</b>\n"
        f"Profit: <b>{metrics['profit_pct']:.2f}%</b> | "
        f"Drawdown: <b>{metrics['drawdown_pct']:.2f}%</b> | "
        f"Win-rate: <b>{metrics['win_rate']:.1f}%</b> | "
        f"Sharpe: <b>{metrics['sharpe']:.2f}</b> | "
        f"Trades: <b>{metrics['num_trades']}</b>\n"
        f"Profit factor: <b>{metrics['profit_factor']:.2f}</b>"
    )
    send_message(txt, parse_mode="HTML", chat_id=chat_id)

    # Alarm-checks – emojis KUN til Telegram
    alarm_msgs = []
    if alert_on_drawdown and check_drawdown_alert(metrics, thresholds.get("drawdown")):
        alarm_msgs.append(
            f"‼️ ADVARSEL: Drawdown ({metrics['drawdown_pct']:.2f}%) under grænsen ({thresholds.get('drawdown', '-')}%)!"
        )
    if alert_on_winrate and check_winrate_alert(metrics, thresholds.get("winrate")):
        alarm_msgs.append(
            f"‼️ ADVARSEL: Win-rate ({metrics['win_rate']:.1f}%) under grænsen ({thresholds.get('winrate', '-')}%)!"
        )
    if alert_on_profit and check_profit_alert(metrics, thresholds.get("profit")):
        alarm_msgs.append(
            f"‼️ ADVARSEL: Profit ({metrics['profit_pct']:.2f}%) under grænsen ({thresholds.get('profit', '-')}%)!"
        )
    for alarm in alarm_msgs:
        send_message(alarm, chat_id=chat_id)

    return metrics


# Eksempel/test
if __name__ == "__main__":
    import pandas as pd

    balance_df = pd.DataFrame({"balance": [1000, 980, 950, 990, 970, 1005]})
    trades_df = pd.DataFrame(
        {
            "type": ["BUY", "TP", "BUY", "SL", "BUY", "TP", "SELL", "TP", "SELL", "SL"],
            "profit": [0, 0.02, 0, -0.015, 0, 0.01, 0, 0.03, 0, -0.012],
        }
    )
    metrics = calculate_live_metrics(trades_df, balance_df)
    print("Live-metrics:", metrics)
    print("Drawdown alarm? ", check_drawdown_alert(metrics))
    print("Win-rate alarm? ", check_winrate_alert(metrics))
    print("Profit alarm? ", check_profit_alert(metrics))
