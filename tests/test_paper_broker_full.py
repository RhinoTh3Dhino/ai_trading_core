# tests/test_paper_broker_full.py
from __future__ import annotations

import csv
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Sørg for at kunne importere brokeren fra projektroden
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytz  # bruges i tidsstempel-test

from bot.brokers.paper_broker import PaperBroker  # noqa: E402


# ---------- små hjælpere ----------
def approx(a: float, b: float, eps: float = 1e-8) -> bool:
    return abs(a - b) <= eps


@dataclass
class TmpLogs:
    fills: Path
    equity: Path
    dir_: tempfile.TemporaryDirectory


def tmp_logs() -> TmpLogs:
    td = tempfile.TemporaryDirectory(prefix="broker_test_")
    base = Path(td.name)
    return TmpLogs(
        fills=base / "fills.csv",
        equity=base / "equity.csv",
        dir_=td,
    )


def read_last_csv_row(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return None
    return rows[-1]


# ---------- TESTS ----------


def test_market_buy_sell_roundtrip():
    logs = tmp_logs()
    b = PaperBroker(
        starting_cash=1000.0,
        commission_bp=0.0,
        slippage_bp=0.0,
        allow_short=False,
        equity_log_path=logs.equity,
        fills_log_path=logs.fills,
    )
    ts0 = datetime(2025, 7, 1, 0, 0, 0)
    b.mark_to_market({"BTCUSDT": 100.0}, ts=ts0)

    # BUY 1 @ 100
    o1 = b.submit_order("BTCUSDT", "BUY", 1.0, "market", ts=ts0)
    assert o1.status == "filled", f"market BUY blev ikke fyldt: {o1}"
    assert approx(b.cash, 900.0)
    assert approx(b.positions["BTCUSDT"].qty, 1.0)
    assert approx(b.positions["BTCUSDT"].avg_price, 100.0)
    assert approx(b.realized_pnl, 0.0)

    # Sælg hele positionen @ 110
    ts1 = ts0 + timedelta(hours=1)
    b.mark_to_market({"BTCUSDT": 110.0}, ts=ts1)
    o2 = b.close_position("BTCUSDT", ts=ts1)
    assert o2 is not None and o2.status == "filled"
    assert approx(b.positions["BTCUSDT"].qty, 0.0)
    assert approx(b.cash, 1010.0), f"cash forventet 1010, fik {b.cash}"
    assert approx(b.realized_pnl, 10.0)

    logs.dir_.cleanup()
    print("OK - market roundtrip uden omkostninger")


def test_no_shorting_rejected():
    logs = tmp_logs()
    b = PaperBroker(
        starting_cash=1000.0,
        commission_bp=0.0,
        slippage_bp=0.0,
        allow_short=False,
        equity_log_path=logs.equity,
        fills_log_path=logs.fills,
    )
    ts = datetime(2025, 7, 1, 0, 0, 0)
    b.mark_to_market({"BTCUSDT": 100.0}, ts=ts)

    # Forsøg at sælge uden long-qty → skal afvises
    o = b.submit_order("BTCUSDT", "SELL", 0.5, "market", ts=ts)
    assert o.status == "rejected" and "Shorting" in (o.reason or "")
    logs.dir_.cleanup()
    print("OK - shorting afvist ved market SELL uden position")


def test_min_filters_submit_and_limit_fill():
    logs = tmp_logs()
    b = PaperBroker(
        starting_cash=10_000.0,
        commission_bp=0.0,
        slippage_bp=0.0,
        min_qty=0.10,
        min_notional=50.0,
        reject_below_min=True,
        equity_log_path=logs.equity,
        fills_log_path=logs.fills,
    )
    ts = datetime(2025, 7, 1, 0, 0, 0)
    b.mark_to_market({"BTCUSDT": 100.0}, ts=ts)

    # For lille qty
    o_small_qty = b.submit_order("BTCUSDT", "BUY", 0.05, "market", ts=ts)
    assert o_small_qty.status == "rejected" and "min_qty" in (o_small_qty.reason or "")

    # For lille notional ved market BUY (men over min_qty)
    # 0.40 * 100 = 40 < 50 → afvises pga. notional
    o_small_notional = b.submit_order("BTCUSDT", "BUY", 0.40, "market", ts=ts)
    assert o_small_notional.status == "rejected" and "Notional" in (o_small_notional.reason or "")

    # Limit BUY med for lille notional ved submit
    o_lim_small = b.submit_order("BTCUSDT", "BUY", 1.0, "limit", limit_price=40.0, ts=ts)
    assert o_lim_small.status == "rejected" and "Notional" in (o_lim_small.reason or "")

    # Valid limit BUY der fyldes når prisen krydser
    o_lim = b.submit_order("BTCUSDT", "BUY", 1.0, "limit", limit_price=60.0, ts=ts)
    assert o_lim.status == "open"
    # pris falder → skal fyldes til bedste pris (≤ limit) = 59
    b.mark_to_market({"BTCUSDT": 59.0}, ts=ts + timedelta(minutes=5))
    pos = b.positions["BTCUSDT"]
    assert approx(pos.qty, 1.0) and approx(pos.avg_price, 59.0)
    assert len(b.open_orders) == 0

    logs.dir_.cleanup()
    print("OK - min_qty/min_notional ved submit og limit-fill")


def test_slippage_and_commission_accounting():
    logs = tmp_logs()
    b = PaperBroker(
        starting_cash=1000.0,
        commission_bp=100.0,  # 1%
        slippage_bp=100.0,  # 1%
        allow_short=False,
        equity_log_path=logs.equity,
        fills_log_path=logs.fills,
        price_decimals=2,
    )
    ts0 = datetime(2025, 7, 1, 0, 0, 0)
    b.mark_to_market({"BTCUSDT": 100.0}, ts=ts0)

    # BUY 1: pris bliver 100*(1+1%)=101; kommission 1% af 101 = 1.01
    o1 = b.submit_order("BTCUSDT", "BUY", 1.0, "market", ts=ts0)
    assert o1.status == "filled"
    assert approx(b.positions["BTCUSDT"].avg_price, 101.0)
    assert approx(b.cash, 1000.0 - 101.0 - 1.01)  # 897.99

    # SELL 1 @ raw 100, slippage 1% ned → 99.0, kommission 0.99
    ts1 = ts0 + timedelta(hours=1)
    b.mark_to_market({"BTCUSDT": 100.0}, ts=ts1)
    o2 = b.close_position("BTCUSDT", ts=ts1)
    assert o2 is not None and o2.status == "filled"
    # Realized PnL = 99 - 101 = -2 (kommissioner i cash, ikke i realized_pnl)
    assert approx(b.realized_pnl, -2.0)
    # Cash = 1000 - 102.01 + 98.01 = 996.0
    assert approx(b.cash, 996.0)

    logs.dir_.cleanup()
    print("OK - slippage og commission bogføres korrekt")


def test_daily_loss_limit_and_reset():
    logs = tmp_logs()
    b = PaperBroker(
        starting_cash=1000.0,
        commission_bp=0.0,
        slippage_bp=0.0,
        daily_loss_limit_pct=1.0,  # 1%
        allow_short=False,
        equity_log_path=logs.equity,
        fills_log_path=logs.fills,
    )
    tz = pytz.timezone("Europe/Copenhagen")
    ts0 = datetime(2025, 7, 1, 10, 0, 0)  # local naive, broker localizer
    b.mark_to_market({"BTCUSDT": 100.0}, ts=ts0)
    b.submit_order("BTCUSDT", "BUY", 1.0, "market", ts=ts0)

    # Fald i pris så equity falder > 1%
    ts1 = ts0 + timedelta(hours=1)
    b.mark_to_market({"BTCUSDT": 90.0}, ts=ts1)  # equity ca. 990 → -1%+ → halt
    assert b.trading_halted is True

    # Ny handel afvises
    o = b.submit_order("BTCUSDT", "BUY", 1.0, "market", ts=ts1)
    assert o.status == "rejected" and "halted" in (o.reason or "").lower()

    # Næste dag → reset af halt
    ts2 = ts0 + timedelta(days=1, hours=1)
    b.mark_to_market({"BTCUSDT": 95.0}, ts=ts2)
    assert b.trading_halted is False

    logs.dir_.cleanup()
    print("OK - daily loss limit trigger og reset næste dag")


def test_fill_timestamp_is_bar_ts_in_csv():
    logs = tmp_logs()
    broker_tz = "Europe/Copenhagen"
    b = PaperBroker(
        starting_cash=1000.0,
        commission_bp=0.0,
        slippage_bp=0.0,
        tz=broker_tz,
        equity_log_path=logs.equity,
        fills_log_path=logs.fills,
    )
    local_tz = pytz.timezone(broker_tz)
    local_bar_time = datetime(2025, 7, 1, 12, 0, 0)  # CEST (UTC+2)
    b.mark_to_market({"BTCUSDT": 100.0}, ts=local_bar_time)
    b.submit_order("BTCUSDT", "BUY", 0.5, "market", ts=local_bar_time)

    # Sidste fill i CSV
    last = read_last_csv_row(logs.fills)
    assert last is not None, "Ingen fills i CSV"
    ts_str = last[0]
    # Forventet Z-tid
    aware = local_tz.localize(local_bar_time).astimezone(pytz.UTC)
    expected = aware.isoformat(timespec="seconds").replace("+00:00", "Z")
    assert ts_str == expected, f"TS i CSV ({ts_str}) matcher ikke bar-ts ({expected})"

    logs.dir_.cleanup()
    print("OK - fill timestamp = barens timestamp (UTC Z) i CSV")


def test_limit_sell_cannot_oversell_no_short():
    logs = tmp_logs()
    b = PaperBroker(
        starting_cash=10_000.0,
        commission_bp=0.0,
        slippage_bp=0.0,
        allow_short=False,
        equity_log_path=logs.equity,
        fills_log_path=logs.fills,
    )
    ts = datetime(2025, 7, 1, 0, 0, 0)
    b.mark_to_market({"BTCUSDT": 100.0}, ts=ts)
    # Først køb 0.4
    b.submit_order("BTCUSDT", "BUY", 0.4, "market", ts=ts)

    # Prøv at oprette limit SELL 0.5 (større end long) → afvises allerede ved submit
    o = b.submit_order("BTCUSDT", "SELL", 0.5, "limit", limit_price=101.0, ts=ts)
    assert o.status == "rejected" and "Shorting" in (o.reason or "")

    # Gyldig SELL 0.4 @ 101 fyldes når pris krydser
    o2 = b.submit_order("BTCUSDT", "SELL", 0.4, "limit", limit_price=101.0, ts=ts)
    assert o2.status == "open"
    b.mark_to_market({"BTCUSDT": 101.0}, ts=ts + timedelta(minutes=1))
    assert approx(b.positions["BTCUSDT"].qty, 0.0)

    logs.dir_.cleanup()
    print("OK - limit SELL kan ikke oversælge (no-short), men gyldig mængde fyldes")


# ---------- simple testrunner ----------


def main():
    tests = [
        test_market_buy_sell_roundtrip,
        test_no_shorting_rejected,
        test_min_filters_submit_and_limit_fill,
        test_slippage_and_commission_accounting,
        test_daily_loss_limit_and_reset,
        test_fill_timestamp_is_bar_ts_in_csv,
        test_limit_sell_cannot_oversell_no_short,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL - {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR - {t.__name__}: {e}")
    if failures == 0:
        print("\n✅ Alle tests bestået.")
    else:
        print(f"\n❌ {failures} test(s) fejlede.")
        sys.exit(1)


if __name__ == "__main__":
    main()
