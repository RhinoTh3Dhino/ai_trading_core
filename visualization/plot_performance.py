# visualization/plot_performance.py

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _choose_equity_col(df: pd.DataFrame) -> str:
    for col in ("balance", "equity", "Balance", "Equity"):
        if col in df.columns:
            return col
    # fallback – lad kaldet håndtere fejl senere
    return "balance"


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series([], dtype="datetime64[ns]")
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series(series), errors="coerce")


def _normalize_drawdown(dd: pd.Series) -> pd.Series:
    """
    Returnér drawdown i procent (negativt), uanset input (decimal eller %).
    - Hvis |max| <= 1 → antag decimal og konverter til procent.
    - Ellers antag allerede procent.
    """
    if dd is None or len(dd) == 0:
        return pd.Series([], dtype=float)
    dd = pd.to_numeric(dd, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if dd.abs().max(skipna=True) <= 1.0:
        dd = dd * 100.0
    return dd.fillna(0.0)


def _compute_drawdown_from_equity(eq: pd.Series) -> pd.Series:
    if eq is None or len(eq) == 0:
        return pd.Series([], dtype=float)
    eq = pd.to_numeric(eq, errors="coerce").fillna(method="ffill").fillna(method="bfill")
    peak = eq.cummax().replace(0, np.nan)
    dd = (eq - peak) / peak
    return dd.fillna(0.0) * 100.0


def _nearest_balance_for_timestamps(
    balance_ts: pd.Series, balance_vals: pd.Series, query_ts: pd.Series
) -> pd.Series:
    """
    For hvert tidspunkt i query_ts, find nærmeste tidsstempel i balance_ts og returnér balance.
    Robust, men O(n log n) – fint til grafer.
    """
    if len(query_ts) == 0 or len(balance_ts) == 0:
        return pd.Series([np.nan] * len(query_ts))
    # merge_asof kræver sortering
    bdf = pd.DataFrame({"ts": balance_ts, "bal": balance_vals}).dropna().sort_values("ts")
    qdf = pd.DataFrame({"ts": query_ts}).dropna().sort_values("ts")
    if bdf.empty or qdf.empty:
        return pd.Series([np.nan] * len(query_ts))
    joined = pd.merge_asof(qdf, bdf, on="ts", direction="nearest", tolerance=pd.Timedelta("3650D"))
    # tilbage til original rækkefølge
    idx_map = pd.Series(range(len(qdf)), index=qdf["ts"])
    order = query_ts.map(idx_map)
    out = joined.loc[order.values, "bal"].reset_index(drop=True)
    # hvis merge_asof ikke fandt noget (NaN), fallback til simpel frem/tilbage udfyld
    return out.fillna(method="ffill").fillna(method="bfill")


def plot_performance(
    balance_df,
    trades_df=None,
    symbol="BTCUSDT",
    model_name=None,
    save_path=None,
    show_trades=True,
    title_extra=None,
    figsize=(14, 7),
    dpi=100,
):
    """
    Plotter equity/balance, drawdown (%) og evt. handler for en AI-model/strategi.

    Args:
        balance_df (pd.DataFrame): Bør indeholde 'timestamp' + 'balance' eller 'equity'.
                                   'drawdown' (decimal eller %) valgfri. 'close' (pris) valgfri.
        trades_df (pd.DataFrame, optional): For markers; bør have 'timestamp' + 'type'. 'balance' valgfri.
        symbol (str): Navn på instrument (fx "BTCUSDT").
        model_name (str): Fx "ML", "DL", "Ensemble".
        save_path (str): Hvor plot gemmes.
        show_trades (bool): Plot BUY/TP/SL-markeringer.
        title_extra (str): Ekstra tekst til titel.
        figsize (tuple): Figur-størrelse.
        dpi (int): Opløsning.

    Returns:
        str: Path til gemt plot.
    """
    if not isinstance(balance_df, pd.DataFrame):
        raise ValueError("balance_df skal være en DataFrame")

    bdf = balance_df.copy()

    # --- Timestamp kolonne ---
    if "timestamp" not in bdf.columns:
        # Prøv 'date' → 'timestamp', ellers brug index
        if "date" in bdf.columns:
            bdf = bdf.rename(columns={"date": "timestamp"})
        else:
            bdf["timestamp"] = bdf.index
    bdf["timestamp"] = _ensure_datetime(bdf["timestamp"])
    bdf = bdf.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # --- Equity/Bal ---
    eq_col = _choose_equity_col(bdf)
    if eq_col not in bdf.columns:
        raise ValueError("balance_df skal indeholde 'balance' eller 'equity'.")
    bdf[eq_col] = (
        pd.to_numeric(bdf[eq_col], errors="coerce").fillna(method="ffill").fillna(method="bfill")
    )

    # --- Drawdown (i %) ---
    if "drawdown" in bdf.columns:
        dd_pct = _normalize_drawdown(bdf["drawdown"])
    else:
        dd_pct = _compute_drawdown_from_equity(bdf[eq_col])
    bdf["__dd_pct__"] = dd_pct

    # --- Plot setup ---
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)

    x = bdf["timestamp"]
    balance = bdf[eq_col]

    # Primær akse: balance/equity
    ax1.plot(x, balance, label="Equity", linewidth=2)
    ax1.set_xlabel("Tid")
    ax1.set_ylabel("Equity", color="black")
    ax1.grid(True, alpha=0.25)

    # Sekundær akse: drawdown i %
    ax3 = ax1.twinx()
    ax3.fill_between(
        x,
        0,
        bdf["__dd_pct__"],
        where=(bdf["__dd_pct__"] < 0),
        alpha=0.18,
        label="Drawdown (%)",
    )
    ax3.set_ylabel("Drawdown (%)")
    # Skub lidt ned så negative områder ses tydeligt
    ymin = min(-1.0, float(bdf["__dd_pct__"].min()) * 1.05 if len(bdf) else -1.0)
    ax3.set_ylim(ymin, 5.0)

    # Pris overlay (på primær akse, skaler separat akse for pris)
    if "close" in bdf.columns:
        ax2 = ax1.twinx()
        ax2.spines.right.set_position(("axes", 1.08))  # flyt tredje akse lidt ud
        ax2.plot(
            x,
            pd.to_numeric(bdf["close"], errors="coerce"),
            linestyle="--",
            linewidth=1,
            alpha=0.35,
            label="Pris",
        )
        ax2.set_ylabel("Pris", color="grey")
        ax2.tick_params(axis="y", labelcolor="grey")

    # --- Trades (BUY/TP/SL) ---
    if show_trades and isinstance(trades_df, pd.DataFrame) and len(trades_df) > 0:
        tdf = trades_df.copy()
        # kræv timestamp
        if "timestamp" not in tdf.columns:
            # hvis 'ts' findes (fx fra paper fills/signals)
            if "ts" in tdf.columns:
                tdf = tdf.rename(columns={"ts": "timestamp"})
            else:
                tdf["timestamp"] = tdf.index
        tdf["timestamp"] = _ensure_datetime(tdf["timestamp"])
        tdf = tdf.dropna(subset=["timestamp"])

        # sørg for en y-værdi til scatter: brug trade.balance hvis tilgængelig, ellers slå nærmeste equity op
        if "balance" in tdf.columns:
            yvals = pd.to_numeric(tdf["balance"], errors="coerce")
            need_lookup = yvals.isna()
        else:
            yvals = pd.Series([np.nan] * len(tdf))
            need_lookup = pd.Series([True] * len(tdf))

        if need_lookup.any():
            looked_up = _nearest_balance_for_timestamps(
                bdf["timestamp"], balance, tdf.loc[need_lookup, "timestamp"]
            )
            yvals.loc[need_lookup] = looked_up.values

        tdf["__y__"] = yvals

        if "type" in tdf.columns:
            types = tdf["type"].astype(str).str.upper()
            buys = tdf[types == "BUY"]
            tps = tdf[types == "TP"]
            sls = tdf[types == "SL"]

            if len(buys):
                ax1.scatter(buys["timestamp"], buys["__y__"], marker="^", label="BUY", zorder=5)
            if len(tps):
                ax1.scatter(tps["timestamp"], tps["__y__"], marker="o", label="TP", zorder=5)
            if len(sls):
                ax1.scatter(sls["timestamp"], sls["__y__"], marker="v", label="SL", zorder=5)

    # --- Titel & legender ---
    title = f"{symbol} | {model_name.upper() if model_name else 'Model'} | Performance"
    if title_extra:
        title += f" | {title_extra}"
    ax1.set_title(title)

    # Saml legender fra flere akser
    handles, labels = [], []
    for ax in (ax1,):
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    # drawdown label fra ax3
    h3, l3 = ax3.get_legend_handles_labels()
    handles += h3
    labels += l3
    if handles:
        ax1.legend(handles, labels, loc="upper left")

    plt.tight_layout()

    # --- Gem plot ---
    if save_path is None:
        os.makedirs("graphs", exist_ok=True)
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Rens modelnavn i filnavn
        mn = (model_name or "model").replace(" ", "_")
        save_path = f"graphs/performance_{symbol}_{mn}_{dt_str}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Gemte performance-plot til: {save_path}")
    return save_path


# === CLI-brug/test ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot performance for AI trading bot")
    parser.add_argument("--balance", type=str, required=True, help="Path til balance_df (CSV)")
    parser.add_argument("--trades", type=str, default=None, help="Path til trades_df (CSV)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--model_name", type=str, default="AI Model")
    parser.add_argument("--title_extra", type=str, default=None)
    args = parser.parse_args()

    balance_df = pd.read_csv(args.balance)
    trades_df = pd.read_csv(args.trades) if args.trades else None
    plot_performance(
        balance_df,
        trades_df=trades_df,
        symbol=args.symbol,
        model_name=args.model_name,
        title_extra=args.title_extra or "CLI Test",
    )
