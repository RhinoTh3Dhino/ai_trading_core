import time
import schedule
import datetime
import traceback


from utils.telegram_utils import send_telegram_message, send_telegram_heartbeat

from utils.project_path import PROJECT_ROOT  # AUTO PATH CONVERTED
def send_daily_status():
    """
    Sender daglig status med trading-metrics.
    Viser profit, win-rate, drawdown, handler mm. fra seneste backtest.
    """
    import pandas as pd
    try:
        df = pd.read_csv(PROJECT_ROOT / "data" / "backtest_results.csv"  # AUTO PATH CONVERTED)
        last = df.iloc[-1]
        msg = (
            f"📊 Daglig status\n"
            f"Profit: {last['profit_pct']}%\n"
            f"Win-rate: {float(last['win_rate'])*100:.1f}%\n"
            f"Drawdown: {last['drawdown_pct']}%\n"
            f"Handler: {last['num_trades']}\n"
            f"Seneste run: {last['timestamp']}"
        )
    except Exception as e:
        msg = f"❌ Kan ikke hente metrics til status ({datetime.datetime.now()})\nFejl: {e}"
    send_telegram_message(msg)

def send_hourly_heartbeat():
    """Sender 'hjertelyd' hver time."""
    send_telegram_heartbeat()

# Planlæg beskeder
schedule.every().day.at("07:00").do(send_daily_status)
schedule.every().hour.at(":00").do(send_hourly_heartbeat)

print("⏰ Scheduler kører! Ctrl+C for at stoppe.")

try:
    while True:
        schedule.run_pending()
        time.sleep(10)   # Sparer CPU
except Exception as e:
    tb = traceback.format_exc()
    send_telegram_message(f"❌ Scheduler/Botten stoppede uventet:\n{e}\n\n{tb}")
    raise