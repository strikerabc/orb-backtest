"""
tools/alpha_funded_combine_analysis.py — ORB pass rates for Alpha Funded challenges.

Simulates each instrument-session combination across all Alpha Funded challenge types.

Usage:
    python tools/alpha_funded_combine_analysis.py --challenge 1step_pro --balance 50000
    python tools/alpha_funded_combine_analysis.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from tools.prop_combine_sim import simulate_combine, CombineRules

_OUT = _ROOT / "outputs"
_OUT.mkdir(exist_ok=True)

# Load Alpha Funded rules
_RULES_PATH = _ROOT / "docs" / "prop_firms" / "alpha_funded_challenges.json"
_AF_RULES = json.loads(_RULES_PATH.read_text(encoding="utf-8"))


def get_challenge_rules(challenge_type: str, balance: int, phase: str = "phase1") -> CombineRules:
    """Convert Alpha Funded challenge spec to CombineRules."""
    forex = _AF_RULES["forex_challenges"]

    # Handle 1-step vs 2-step
    if challenge_type in ["1step_lite", "1step_pro"]:
        spec = forex[challenge_type][phase]
        target_pct = spec["profit_target_pct"]
        dd_pct = spec["max_loss_pct"]
    elif challenge_type in ["2step_standard", "2step_pro"]:
        spec = forex[challenge_type][phase]
        target_pct = spec["profit_target_pct"]
        dd_pct = spec["max_loss_pct"]
    elif challenge_type in ["instant_funding", "instant_funding_pro"]:
        # Instant funding has no phases, go straight to funded rules
        spec = forex[challenge_type]["funded"]
        # No profit target for instant, simulate 6% as baseline (same as 1step_pro)
        target_pct = 6.0
        dd_pct = spec["max_loss_pct"]
    else:
        raise ValueError(f"Unknown challenge: {challenge_type}")

    return CombineRules(
        account_size=balance,
        profit_target=int(balance * target_pct / 100),
        max_drawdown=int(balance * dd_pct / 100),
        trailing_dd=True  # All Alpha Funded forex challenges use trailing DD
    )


def main() -> None:
    p = argparse.ArgumentParser(description="ORB pass rates for Alpha Funded challenges")
    p.add_argument("--challenge", choices=[
        "1step_lite", "1step_pro", "2step_standard", "2step_pro",
        "instant_funding", "instant_funding_pro"
    ], help="Challenge type")
    p.add_argument("--balance", type=int, choices=[10000, 25000, 50000, 100000, 200000, 300000, 400000],
                   help="Account balance")
    p.add_argument("--phase", choices=["phase1", "phase2"], default="phase1",
                   help="Phase (2-step only)")
    p.add_argument("--all", action="store_true", help="Run all challenge types at all balances")
    p.add_argument("--risk-pct", type=float, default=0.5,
                   help="Risk per trade as %% of account (default: 0.5%%)")
    p.add_argument("--n-sims", type=int, default=10_000, help="Monte Carlo simulations")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    args = p.parse_args()

    if not args.all and (not args.challenge or not args.balance):
        p.error("Either --all or both --challenge and --balance required")

    # Load ORB edges
    edges_path = _ROOT / "outputs" / "hyp01_gross_net_by_instrument_session.csv"
    if not edges_path.exists():
        print(f"ERROR: {edges_path} not found. Run HYP-01 analysis first.")
        sys.exit(1)

    edges = pd.read_csv(edges_path)
    # Rename columns to match expected format
    edges = edges.rename(columns={"session": "session_name", "E_net_r": "mean_r_net"})

    # Run simulation
    if args.all:
        run_all_challenges(edges, args)
    else:
        run_single_challenge(edges, args)


def run_single_challenge(edges: pd.DataFrame, args) -> None:
    """Simulate a single challenge type."""
    rules = get_challenge_rules(args.challenge, args.balance, args.phase)

    print(f"\n{'=' * 100}")
    print(f"ALPHA FUNDED: {args.challenge.upper().replace('_', ' ')} - ${args.balance:,}")
    if args.challenge.startswith("2step") and args.phase == "phase2":
        print(f"PHASE 2")
    print(f"{'=' * 100}")
    print(f"Profit target: ${rules.profit_target:,} ({100*rules.profit_target/rules.starting_balance:.1f}%)")
    print(f"Max DD: ${rules.max_drawdown:,} ({100*rules.max_drawdown/rules.starting_balance:.1f}%, trailing)")
    print(f"Risk/trade: {args.risk_pct}% of account")
    print(f"Simulations: {args.n_sims:,}\n")

    print(f"{'Instrument':<12} {'Session':<8} {'n':<6} {'Edge R':<10} {'Pass%':<8}")
    print("-" * 100)

    results = []
    for _, row in edges.iterrows():
        sym = row["instrument"]
        sess = row["session_name"]
        n = int(row["n_trades"])
        edge = float(row["mean_r_net"])

        if n < 10:
            continue

        # Estimate win rate from edge (assume 1.5:1 R:R)
        estimated_wr = (edge + 1.0) / 2.5
        estimated_wr = max(0.01, min(0.99, estimated_wr))

        sim_result = simulate_combine(
            edge_per_trade=edge,
            win_rate=estimated_wr,
            risk_per_trade_pct=args.risk_pct,
            rules=rules,
            n_sims=args.n_sims,
            seed=args.seed
        )

        pass_pct = sim_result["pass_pct"]
        print(f"{sym:<12} {sess:<8} {n:<6} {edge:+.4f}    {pass_pct:6.1f}%")

        results.append({
            "instrument": sym,
            "session_name": sess,
            "n_trades": n,
            "mean_r_net": edge,
            "pass_pct": pass_pct,
        })

    mean_pass = np.mean([r["pass_pct"] for r in results])
    median_pass = np.median([r["pass_pct"] for r in results])

    print(f"\n{'=' * 100}")
    print(f"SUMMARY")
    print(f"{'=' * 100}")
    print(f"Mean pass rate: {mean_pass:.1f}%")
    print(f"Median pass rate: {median_pass:.1f}%")

    out_file = _OUT / f"alpha_funded_{args.challenge}_{args.balance}_{args.phase}.json"
    out_file.write_text(json.dumps({
        "challenge": args.challenge,
        "balance": args.balance,
        "phase": args.phase,
        "profit_target": rules.profit_target,
        "max_drawdown": rules.max_drawdown,
        "risk_pct": args.risk_pct,
        "n_sims": args.n_sims,
        "mean_pass_pct": round(mean_pass, 2),
        "median_pass_pct": round(median_pass, 2),
        "results": results,
    }, indent=2), encoding="utf-8")

    print(f"\nResults saved to {out_file}")


def run_all_challenges(edges: pd.DataFrame, args) -> None:
    """Run all challenge types at all balances and generate comparison table."""
    print(f"\n{'=' * 120}")
    print(f"ALPHA FUNDED: ALL CHALLENGES — ORB MEDIAN PASS RATES")
    print(f"{'=' * 120}\n")

    balances = [50000, 100000, 200000]  # Focus on common sizes
    challenges = [
        ("1step_lite", "phase1"),
        ("1step_pro", "phase1"),
        ("2step_standard", "phase1"),
        ("2step_standard", "phase2"),
        ("2step_pro", "phase1"),
        ("2step_pro", "phase2"),
    ]

    summary = []

    for challenge, phase in challenges:
        row_data = {"challenge": f"{challenge}_{phase}"}

        for balance in balances:
            rules = get_challenge_rules(challenge, balance, phase)

            combo_passes = []
            for _, row in edges.iterrows():
                n = int(row["n_trades"])
                edge = float(row["mean_r_net"])

                if n < 10:
                    continue

                estimated_wr = (edge + 1.0) / 2.5
                estimated_wr = max(0.01, min(0.99, estimated_wr))

                sim_result = simulate_combine(
                    edge_per_trade=edge,
                    win_rate=estimated_wr,
                    risk_per_trade_pct=args.risk_pct,
                    rules=rules,
                    n_sims=args.n_sims,
                    seed=args.seed
                )

                combo_passes.append(sim_result["pass_pct"])

            median_pass = np.median(combo_passes)
            row_data[f"${balance//1000}k"] = f"{median_pass:.1f}%"

        summary.append(row_data)

    # Print table
    print(f"{'Challenge':<30} {'$50k':<10} {'$100k':<10} {'$200k':<10}")
    print("-" * 120)
    for row in summary:
        ch = row["challenge"].replace("_", " ").replace("phase1", "P1").replace("phase2", "P2")
        print(f"{ch:<30} {row.get('$50k', 'N/A'):<10} {row.get('$100k', 'N/A'):<10} {row.get('$200k', 'N/A'):<10}")

    out_file = _OUT / "alpha_funded_all_challenges_summary.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {out_file}")


if __name__ == "__main__":
    main()
