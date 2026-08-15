"""
orb_regime_analysis.py — Analyze why holdout failed (2024+) vs in-sample (2021-2023).

Compare market conditions, ATR distributions, trade characteristics, exit reasons.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data_layer import ensure_data
from src.range_builder import build_session_days
from src.entry_detector import detect_entries

INSTRUMENT = "NQ"
SESSION = "NY"
RANGE_MINS = 5
CLOSURE_TF = 5
BASE_RR = 0.5
BREAKEVEN_RR = 0.25
TRAIL_START_RR = 0.5

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-01-01"
HOLDOUT_START = "2024-01-01"

# ATR 1.0× winner from previous test
TRAIL_METHOD = "atr"
TRAIL_PARAM = 1.0


def simulate_trade(
    df_1m: pd.DataFrame,
    entry_ts: pd.Timestamp,
    entry_price: float,
    direction: str,
    stop_initial: float,
    target_initial: float,
    atr_at_entry: float,
) -> dict:
    """Simulate one trade with ATR 1.0× trailing, return detailed outcome."""
    df_post = df_1m[df_1m["timestamp"] > entry_ts].copy()
    if len(df_post) == 0:
        return {
            "gross_r": 0.0, "exit_reason": "eod", "bars_held": 0,
            "max_fav_r": 0.0, "max_adv_r": 0.0
        }

    risk = abs(entry_price - stop_initial)
    highs = df_post["high"].to_numpy()
    lows = df_post["low"].to_numpy()

    if direction == "long":
        fav_prices = highs
        adv_prices = lows
        fav_r = (fav_prices - entry_price) / risk
        adv_r = (lows - entry_price) / risk
    else:
        fav_prices = lows
        adv_prices = highs
        fav_r = (entry_price - fav_prices) / risk
        adv_r = (entry_price - highs) / risk

    stop_current = stop_initial
    target_active = True
    trailing_active = False
    max_fav = 0.0
    max_adv = 0.0

    for i in range(len(df_post)):
        fav_p = fav_prices[i]
        adv_p = adv_prices[i]
        fav_r_i = fav_r[i]
        adv_r_i = adv_r[i]

        max_fav = max(max_fav, fav_r_i)
        max_adv = min(max_adv, adv_r_i)

        # Check stop
        if direction == "long":
            stop_hit = adv_p <= stop_current
        else:
            stop_hit = adv_p >= stop_current

        # Check target
        if target_active:
            if direction == "long":
                target_hit = fav_p >= target_initial
            else:
                target_hit = fav_p <= target_initial
        else:
            target_hit = False

        if stop_hit:
            exit_r = (stop_current - entry_price) / risk if direction == "long" else (entry_price - stop_current) / risk
            reason = "trail_stop" if trailing_active else ("breakeven" if stop_current == entry_price else "stop")
            return {
                "gross_r": float(exit_r), "exit_reason": reason, "bars_held": i + 1,
                "max_fav_r": float(max_fav), "max_adv_r": float(max_adv)
            }

        if target_hit:
            exit_r = (target_initial - entry_price) / risk if direction == "long" else (entry_price - target_initial) / risk
            return {
                "gross_r": float(exit_r), "exit_reason": "target", "bars_held": i + 1,
                "max_fav_r": float(max_fav), "max_adv_r": float(max_adv)
            }

        # Breakeven
        if fav_r_i >= BREAKEVEN_RR and stop_current == stop_initial:
            stop_current = entry_price
            target_active = False

        # Start trailing
        if fav_r_i >= TRAIL_START_RR and not trailing_active:
            trailing_active = True

        # Trail with ATR 1.0×
        if trailing_active:
            trail_dist = TRAIL_PARAM * atr_at_entry
            if direction == "long":
                new_stop = fav_p - trail_dist
                stop_current = max(stop_current, new_stop)
            else:
                new_stop = fav_p + trail_dist
                stop_current = min(stop_current, new_stop)

    # EOD
    last = df_post.iloc[-1]
    exit_r = (last["close"] - entry_price) / risk if direction == "long" else (entry_price - last["close"]) / risk
    return {
        "gross_r": float(exit_r), "exit_reason": "eod", "bars_held": len(df_post),
        "max_fav_r": float(max_fav), "max_adv_r": float(max_adv)
    }


def main():
    print("=" * 100)
    print("REGIME ANALYSIS: In-Sample (2021-2023) vs Holdout (2024+)")
    print("=" * 100)

    print("\n1. Loading data...")
    df_1m = ensure_data(INSTRUMENT)

    print("\n2. Building sessions...")
    session_days = build_session_days(df_1m, INSTRUMENT, SESSION)

    print("\n3. Detecting entries...")
    all_signals = []
    signal_to_sd = {}
    for sd in session_days:
        signals = detect_entries(sd)
        for sig in signals:
            all_signals.append(sig)
            signal_to_sd[id(sig)] = sd

    target_signals = [
        s for s in all_signals
        if s.mode == "CC" and s.range_minutes == RANGE_MINS and s.closure_tf == CLOSURE_TF
    ]

    # Build entries
    entries_data = []
    for sig in target_signals:
        sd = signal_to_sd[id(sig)]
        if sig.entry_bar_idx >= len(sd.bar_timestamps):
            continue

        bar_ts = pd.Timestamp(sd.bar_timestamps[sig.entry_bar_idx], tz="UTC")
        risk = abs(sig.fill_price - sig.sl_price)
        target = sig.fill_price + (BASE_RR * risk) if sig.direction == "long" else sig.fill_price - (BASE_RR * risk)

        entries_data.append({
            "timestamp": bar_ts,
            "trigger_price": sig.fill_price,
            "direction": sig.direction,
            "stop_price": sig.sl_price,
            "target_price": target,
            "atr_4h": sd.atr_4h,
            "date": bar_ts.date(),
        })

    entries_df = pd.DataFrame(entries_data)

    # Split periods
    is_df = entries_df[(entries_df["timestamp"] >= IN_SAMPLE_START) & (entries_df["timestamp"] < IN_SAMPLE_END)].copy()
    ho_df = entries_df[entries_df["timestamp"] >= HOLDOUT_START].copy()

    print(f"\n   In-sample: {len(is_df)} trades")
    print(f"   Holdout: {len(ho_df)} trades")

    # Simulate all trades
    print("\n4. Simulating trades...")
    for period_name, period_df in [("in_sample", is_df), ("holdout", ho_df)]:
        outcomes = []
        for idx, ent in period_df.iterrows():
            out = simulate_trade(
                df_1m, ent["timestamp"], ent["trigger_price"],
                ent["direction"], ent["stop_price"], ent["target_price"],
                ent["atr_4h"]
            )
            outcomes.append(out)

        period_df["gross_r"] = [o["gross_r"] for o in outcomes]
        period_df["exit_reason"] = [o["exit_reason"] for o in outcomes]
        period_df["bars_held"] = [o["bars_held"] for o in outcomes]
        period_df["max_fav_r"] = [o["max_fav_r"] for o in outcomes]
        period_df["max_adv_r"] = [o["max_adv_r"] for o in outcomes]

    # ── ANALYSIS ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("COMPARATIVE ANALYSIS")
    print("=" * 100)

    def analyze_period(df: pd.DataFrame, name: str) -> dict:
        returns = df["gross_r"].to_numpy()
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0.0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0.0

        exit_dist = df["exit_reason"].value_counts(normalize=True).to_dict()

        return {
            "n": len(df),
            "mean_r": float(returns.mean()),
            "std_r": float(returns.std()),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "median_r": float(np.median(returns)),
            "atr_mean": float(df["atr_4h"].mean()),
            "atr_std": float(df["atr_4h"].std()),
            "bars_held_mean": float(df["bars_held"].mean()),
            "max_fav_r_mean": float(df["max_fav_r"].mean()),
            "max_adv_r_mean": float(df["max_adv_r"].mean()),
            "exit_reasons": exit_dist,
            "long_pct": float((df["direction"] == "long").mean()),
        }

    is_stats = analyze_period(is_df, "In-Sample")
    ho_stats = analyze_period(ho_df, "Holdout")

    print(f"\n{'Metric':<25} {'In-Sample':<20} {'Holdout':<20} {'Change':<15}")
    print("-" * 80)

    metrics = [
        ("Trades", "n", "d"),
        ("Mean R", "mean_r", ".4f"),
        ("Std R", "std_r", ".4f"),
        ("Win Rate", "win_rate", ".2%"),
        ("Avg Win", "avg_win", ".4f"),
        ("Avg Loss", "avg_loss", ".4f"),
        ("Median R", "median_r", ".4f"),
        ("ATR Mean (pts)", "atr_mean", ".2f"),
        ("ATR Std", "atr_std", ".2f"),
        ("Bars Held (avg)", "bars_held_mean", ".1f"),
        ("Max Fav R (avg)", "max_fav_r_mean", ".3f"),
        ("Max Adv R (avg)", "max_adv_r_mean", ".3f"),
        ("Long %", "long_pct", ".2%"),
    ]

    for label, key, fmt in metrics:
        is_val = is_stats[key]
        ho_val = ho_stats[key]
        if fmt == ".2%":
            is_str = f"{is_val:.2%}"
            ho_str = f"{ho_val:.2%}"
            delta = f"{(ho_val - is_val):.2%}"
        elif fmt == ".4f":
            is_str = f"{is_val:+.4f}"
            ho_str = f"{ho_val:+.4f}"
            delta = f"{(ho_val - is_val):+.4f}"
        elif fmt == ".3f":
            is_str = f"{is_val:+.3f}"
            ho_str = f"{ho_val:+.3f}"
            delta = f"{(ho_val - is_val):+.3f}"
        elif fmt == ".2f":
            is_str = f"{is_val:.2f}"
            ho_str = f"{ho_val:.2f}"
            delta = f"{(ho_val - is_val):+.2f}"
        elif fmt == ".1f":
            is_str = f"{is_val:.1f}"
            ho_str = f"{ho_val:.1f}"
            delta = f"{(ho_val - is_val):+.1f}"
        else:  # int
            is_str = f"{is_val}"
            ho_str = f"{ho_val}"
            delta = f"{(ho_val - is_val):+d}"

        print(f"{label:<25} {is_str:<20} {ho_str:<20} {delta:<15}")

    print("\n" + "=" * 100)
    print("EXIT REASON DISTRIBUTION")
    print("=" * 100)
    print(f"\n{'Exit Reason':<20} {'In-Sample':<15} {'Holdout':<15}")
    print("-" * 50)

    all_reasons = set(is_stats["exit_reasons"].keys()) | set(ho_stats["exit_reasons"].keys())
    for reason in sorted(all_reasons):
        is_pct = is_stats["exit_reasons"].get(reason, 0.0)
        ho_pct = ho_stats["exit_reasons"].get(reason, 0.0)
        print(f"{reason:<20} {is_pct:<15.2%} {ho_pct:<15.2%}")

    # Monthly breakdown
    print("\n" + "=" * 100)
    print("MONTHLY PERFORMANCE (Holdout)")
    print("=" * 100)

    ho_df["month"] = pd.to_datetime(ho_df["timestamp"]).dt.to_period("M")
    monthly = ho_df.groupby("month")["gross_r"].agg(["count", "mean", "sum"]).reset_index()
    monthly.columns = ["Month", "Trades", "Mean R", "Cumulative R"]
    monthly["Month"] = monthly["Month"].astype(str)  # Convert Period to string for JSON
    print("\n" + monthly.to_string(index=False))

    # Save results
    output = {
        "in_sample": is_stats,
        "holdout": ho_stats,
        "monthly_holdout": monthly.to_dict(orient="records"),
    }

    out_path = _ROOT / "outputs" / "orb_regime_analysis.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n   Results: {out_path}")


if __name__ == "__main__":
    main()
