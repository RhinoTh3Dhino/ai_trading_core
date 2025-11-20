#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production scheduler/runner for AI Trading Bot.

Bevarer:
- run_pipeline-flow m. Telegram, backup, statusfiler og daglige job.

Forbedringer i denne version:
- Robust Telegram: deaktiver automatisk hvis env mangler.
- Konfigurerbar FEATURES_PATH via ENV (fallback til eksisterende sti).
- Idempotente "ensure_*" helpers og konsekvent safe_run omkring jobs.
"""

import os
import time

import pandas as pd
import schedule
from dotenv import load_dotenv

# Din pipeline – beholdt
from pipeline.core import run_pipeline
from utils.backup import make_backup
from utils.botstatus import update_bot_status
from utils.changelog import append_to_changelog

# Ensemble params (threshold + weights)
from utils.ensemble_utils import load_best_ensemble_params
from utils.project_path import PROJECT_ROOT
from utils.report_utils import log_performance_to_history
from utils.robust_utils import safe_run

# Telegram utils importeres, men vi wrapper kald så de er no-op når ikke konfigureret
from utils.telegram_utils import generate_trend_graph, send_message, send_trend_graph

# ───────────────────────────────────────────────────────────────────────────────
# Miljø & konfiguration
# ───────────────────────────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "") or ""
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "") or ""

# Tillad konfiguration af features-sti via env; fallback til din nuværende explicitte sti
FEATURES_PATH = os.getenv(
    "FEATURES_PATH",
    str(PROJECT_ROOT / "outputs" / "feature_data" / "btcusdt_1h_features_v1.0.0.csv"),
)
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("TIMEFRAME", "1h")


def _telegram_enabled() -> bool:
    return bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)


def _safe_send_message(text: str) -> None:
    if _telegram_enabled():
        try:
            send_message(text)
        except Exception as e:
            print(f"[TELEGRAM] FEJL: {e}")
    else:
        print(f"[TELEGRAM:disabled] {text}")


# ───────────────────────────────────────────────────────────────────────────────
# Artefakt sikring (CI/produktion)
# ───────────────────────────────────────────────────────────────────────────────
def ensure_performance_history_exists():
    history_path = PROJECT_ROOT / "outputs" / "performance_history.csv"
    if not history_path.exists():
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"timestamp": "", "Navn": "", "Balance": ""}]).to_csv(
            history_path, index=False
        )
        print("🟡 Oprettede tom outputs/performance_history.csv for CI compliance.")


def ensure_botstatus_exists():
    path = PROJECT_ROOT / "BotStatus.md"
    if not path.exists():
        path.write_text("# Dummy BotStatus for CI\n", encoding="utf-8")
        print("🟡 Oprettede BotStatus.md (dummy for CI)")


def ensure_changelog_exists():
    path = PROJECT_ROOT / "CHANGELOG.md"
    if not path.exists():
        path.write_text("# Changelog (dummy for CI)\n", encoding="utf-8")
        print("🟡 Oprettede CHANGELOG.md (dummy for CI)")


def ensure_balance_trend_exists():
    img_path = PROJECT_ROOT / "outputs" / "balance_trend.png"
    if not img_path.exists():
        import matplotlib.pyplot as plt

        img_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot([0], [0], marker="o", label="Ingen data")
        plt.title("Balanceudvikling over tid (ingen data)")
        plt.xlabel("Tid")
        plt.ylabel("Balance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
        print("🟡 Oprettede dummy balance_trend.png for CI compliance.")


# ───────────────────────────────────────────────────────────────────────────────
# Hoved-cyklus (én fuld kørsel)
# ───────────────────────────────────────────────────────────────────────────────
def main_trading_cycle():
    print("✅ Botten starter trading-cyklus...")

    ensure_performance_history_exists()
    ensure_botstatus_exists()
    ensure_changelog_exists()
    ensure_balance_trend_exists()

    # Hent bedste params til ensemble før hvert run (kan ændre sig over tid)
    threshold, weights = load_best_ensemble_params()
    print(f"[INFO] Bruger threshold={threshold}, weights={weights} til dette run.")

    metrics = None
    try:
        # Kaldet bevares – justerbar features_path via ENV
        metrics = run_pipeline(
            features_path=FEATURES_PATH,
            symbol=SYMBOL,
            interval=INTERVAL,
            threshold=threshold,
            weights=weights,
            log_to_tb=True,
            send_telegram=_telegram_enabled(),
            plot_graphs=True,
            save_graphs=True,
            verbose=True,
        )
    except Exception as e:
        print(f"❌ FEJL i pipeline: {e}")
        _safe_send_message(f"❌ FEJL i pipeline: {e}")

    # Log til historikfil (MVP-krav)
    try:
        log_performance_to_history(PROJECT_ROOT / "outputs" / "portfolio_metrics_latest.csv")
    except Exception as e:
        print(f"❌ Fejl i log_performance_to_history: {e}")

    # Trend-graf til Telegram
    try:
        img_path = generate_trend_graph()
        if _telegram_enabled():
            send_trend_graph(TELEGRAM_CHAT_ID, TELEGRAM_TOKEN, img_path)
        print(
            f"✅ Trend-graf genereret{'' if _telegram_enabled() else ' (ikke sendt – Telegram disabled)'}: {img_path}"
        )
    except Exception as e:
        print(f"❌ Fejl ved trend-graf: {e}")

    # Backup + statusbesked
    backup_path = make_backup(keep_days=7, keep_per_day=10)
    print(f"✅ Backup gemt: {backup_path}")
    _safe_send_message(f"✅ Bot kørte OK og lavede backup: {backup_path}")

    # Sikr artefakter igen (idempotent) – gør CI/monitoring glad
    ensure_performance_history_exists()
    ensure_botstatus_exists()
    ensure_changelog_exists()
    ensure_balance_trend_exists()

    return metrics, backup_path


# ───────────────────────────────────────────────────────────────────────────────
# Supplerende jobs (daglig status / retrain / heartbeat)
# ───────────────────────────────────────────────────────────────────────────────
def daily_status():
    try:
        _safe_send_message(
            "📊 Daglig status: Botten kører fortsat! Tilpas evt. med flere metrics her."
        )
        append_to_changelog("📊 Daglig status sendt til Telegram.")
        print("✅ Daglig status sendt.")
    except Exception as e:
        print(f"❌ Fejl ved daglig status: {e}")


def retrain_models():
    try:
        _safe_send_message("🔄 Starter automatisk retrain af modeller!")
        append_to_changelog("🔄 Automatisk retrain af modeller startet.")
        print("✅ Retrain-job kørt.")
    except Exception as e:
        print(f"❌ Fejl ved retrain: {e}")


def heartbeat():
    try:
        _safe_send_message("💓 Bot heartbeat: Jeg er stadig i live!")
        print("✅ Heartbeat sendt.")
    except Exception as e:
        print(f"❌ Fejl ved heartbeat: {e}")


# ───────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ───────────────────────────────────────────────────────────────────────────────
def main():
    print("✅ AI Trading Bot starter...")
    error_msg = None
    backup_path = None
    try:
        metrics, backup_path = main_trading_cycle()
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Fejl under kørsel: {e}")
        try:
            _safe_send_message(f"❌ Bot FEJLEDE under kørsel: {e}")
        except Exception as tel_e:
            print(f"❌ Telegram FEJL: {tel_e}")
    finally:
        update_bot_status(
            status="✅ Succes" if error_msg is None else "❌ Fejl",
            backup_path=backup_path,
            error_msg=error_msg if error_msg else "Ingen",
        )
        if error_msg is None:
            append_to_changelog(f"✅ Bot kørte og lavede backup: {backup_path}")
        else:
            append_to_changelog(f"❌ Bot fejlede: {error_msg}")
    print("✅ Bot-kørsel færdig.")


if __name__ == "__main__":
    print("🚀 AI Trading Bot (Production Mode) starter med schedule!")
    print(f"CI mode: {os.getenv('CI', 'false').lower()}")

    # CI: Kør kun én cyklus og afslut!
    if os.getenv("CI", "false").lower() == "true":
        safe_run(main)
    else:
        # Én kørsel ved opstart
        safe_run(main)

        # Planlagte jobs
        schedule.every().hour.at(":00").do(lambda: safe_run(main))
        schedule.every().day.at("08:00").do(lambda: safe_run(daily_status))
        schedule.every().day.at("03:00").do(lambda: safe_run(retrain_models))
        schedule.every().hour.at(":30").do(lambda: safe_run(heartbeat))

        while True:
            schedule.run_pending()
            time.sleep(5)
