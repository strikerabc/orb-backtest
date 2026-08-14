"""
tools/orb_combine_analysis.py — Compute prop firm combine pass rates for all ORB
instrument-session combinations.

Loads trade cache, computes mean R/trade per instrument-session, simulates combine
pass rates under specified firm rules.

Usage:
    python tools/orb_combine_analysis.py --firm alphafunded --challenge standard
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from tools.prop_combine_sim import simulate_combine, CombineRules

_CACHE = _ROOT / "data" / "cache_trade_universe.parquet"
_OUT = _ROOT / "outputs"


def load_orb_edges() -> pd.DataFrame:
    """
    Load ORB per-instrument-session edges from HYP-01 output.

    Returns DataFrame with columns:
        instrument, session_name, n_trades, mean_r_net
    """
    csv_path = _OUT / "hyp01_gross_net_by_instrument_session.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HYP-01 output not found at {csv_path}. "
            "Run tools/analysis/hyp01_openrange.py first."
        )

    df = pd.read_csv(csv_path)

    # Rename columns to match expected format
    df = df.rename(columns={
        "session": "session_name",
        "E_net_r": "mean_r_net"
    })

    return df[["instrument", "session_name", "n_trades", "mean_r_net"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-size", type=float, default=50_000)
    parser.add_argument("--profit-target", type=float, default=3_000)
    parser.add_argument("--max-dd", type=float, default=2_000)
    parser.add_argument("--trailing", action="store_true", default=True)
    parser.add_argument("--risk-pct", type=float, default=0.5,
                        help="Risk per trade as percent of account")
    parser.add_argument("--n-sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("ORB COMBINE PASS RATE ANALYSIS")
    print("=" * 100)
    print(f"Account: ${args.account_size:,.0f}")
    print(f"Profit target: ${args.profit_target:,.0f}")
    print(f"Max DD: ${args.max_dd:,.0f} ({'trailing' if args.trailing else 'static'})")
    print(f"Risk/trade: {args.risk_pct}% of account")
    print(f"Simulations: {args.n_sims:,}")

    # Load ORB edges
    print("\nLoading ORB trade data...")
    edges = load_orb_edges()
    print(f"Found {len(edges)} instrument-session combinations")

    # Define combine rules
    rules = CombineRules(
        account_size=args.account_size,
        profit_target=args.profit_target,
        max_drawdown=args.max_dd,
        trailing_dd=args.trailing,
    )

    # Simulate each instrument-session
    print("\n" + "=" * 100)
    print("COMBINE PASS RATES BY INSTRUMENT-SESSION")
    print("=" * 100)
    print(f"{'Instrument':<12} {'Session':<8} {'n':<6} {'Edge R':<10} {'Pass%':<8} {'Fail%':<8}")
    print("-" * 100)

    results = []
    for _, row in edges.iterrows():
        sym = row["instrument"]
        sess = row["session_name"]
        n = int(row["n_trades"])
        edge = float(row["mean_r_net"])

        if n < 10:  # skip combos with very few trades
            continue

        # For simulation, we need win_rate. Estimate from edge assuming 1.5:1 R:R
        # At 1.5:1, edge = wr * 1.5 + (1-wr) * (-1) = 2.5*wr - 1
        # So wr = (edge + 1) / 2.5
        estimated_wr = (edge + 1.0) / 2.5
        estimated_wr = max(0.01, min(0.99, estimated_wr))  # clamp to valid range

        # Simulate combine
        sim_result = simulate_combine(
            edge_per_trade=edge,
            win_rate=estimated_wr,
            risk_per_trade_pct=args.risk_pct,
            rules=rules,
            n_sims=args.n_sims,
            seed=args.seed
        )

        pass_pct = sim_result["pass_pct"]
        fail_pct = sim_result["fail_pct"]

        print(f"{sym:<12} {sess:<8} {n:<6} {edge:+.4f}    {pass_pct:6.1f}%  {fail_pct:6.1f}%")

        results.append({
            "instrument": sym,
            "session_name": sess,
            "n_trades": n,
            "mean_r_net": edge,
            "pass_pct": pass_pct,
            "fail_pct": fail_pct,
            "median_trades": sim_result["median_trades"],
        })

    # Summary stats
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    pass_rates = [r["pass_pct"] for r in results]
    print(f"Mean pass rate across all combos: {np.mean(pass_rates):.1f}%")
    print(f"Median pass rate: {np.median(pass_rates):.1f}%")
    print(f"Best combo: {max(pass_rates):.1f}%")
    print(f"Worst combo: {min(pass_rates):.1f}%")

    # Save results
    output = {
        "rules": {
            "account_size": args.account_size,
            "profit_target": args.profit_target,
            "max_drawdown": args.max_dd,
            "trailing_dd": args.trailing,
            "risk_per_trade_pct": args.risk_pct,
        },
        "n_sims": args.n_sims,
        "seed": args.seed,
        "instrument_sessions": results,
        "summary": {
            "mean_pass_pct": float(np.mean(pass_rates)),
            "median_pass_pct": float(np.median(pass_rates)),
            "best_pass_pct": float(max(pass_rates)),
            "worst_pass_pct": float(min(pass_rates)),
        }
    }

    out_path = _OUT / "orb_combine_analysis.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
