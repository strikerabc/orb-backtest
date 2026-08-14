"""
tools/prop_combine_sim.py — Monte Carlo simulation of prop firm combine pass rates.

Given edge distribution (mean R/trade, win rate at specific R:R), simulate N combine
attempts and compute pass/fail/timeout rates under specific firm rules.

Usage:
    from tools.prop_combine_sim import simulate_combine, CombineRules

    rules = CombineRules(
        account_size=50_000,
        profit_target=3_000,
        max_drawdown=2_000,
        trailing_dd=True,
        time_limit_days=None,
        min_trading_days=None
    )

    result = simulate_combine(
        edge_per_trade=0.10,
        win_rate=0.44,
        risk_per_trade_pct=0.5,
        rules=rules,
        n_sims=10_000,
        seed=0
    )

    print(f"Pass rate: {result['pass_pct']:.1f}%")
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CombineRules:
    """Prop firm combine challenge rules."""
    account_size: float
    profit_target: float  # $ amount to reach
    max_drawdown: float   # $ amount max loss from starting balance or peak
    trailing_dd: bool = False  # True = trailing from peak, False = static from start
    time_limit_days: Optional[int] = None  # None = unlimited
    min_trading_days: Optional[int] = None  # minimum days with at least 1 trade

    def __post_init__(self):
        self.profit_target_r = self.profit_target / self.max_drawdown
        self.max_dd_r = self.max_drawdown / self.max_drawdown  # = 1.0 R by definition


def simulate_combine(
    edge_per_trade: float,
    win_rate: float,
    risk_per_trade_pct: float,
    rules: CombineRules,
    n_sims: int = 10_000,
    seed: int = 0,
) -> dict:
    """
    Simulate combine pass rate via Monte Carlo.

    Args:
        edge_per_trade: Mean R/trade (e.g. 0.10 = +0.10R expectancy)
        win_rate: Probability of winning trade (e.g. 0.44)
        risk_per_trade_pct: Risk per trade as % of account (e.g. 0.5 = 0.5%)
        rules: CombineRules specifying challenge geometry
        n_sims: Number of Monte Carlo runs
        seed: RNG seed

    Returns:
        dict with keys: pass_pct, fail_pct, timeout_pct, median_trades,
                        pass_count, fail_count, timeout_count
    """
    rng = np.random.default_rng(seed)

    # Compute win/loss R values from edge and win rate
    # edge = p_win * R_win + (1 - p_win) * R_loss
    # For simplicity, assume R:R ratio = 1.5:1 (risking 1R to make 1.5R)
    # Then R_win = +1.5, R_loss = -1.0
    # edge = p_win * 1.5 + (1 - p_win) * (-1.0) = 2.5 * p_win - 1.0
    # Solving for p_win: p_win = (edge + 1.0) / 2.5
    #
    # But user provided win_rate directly, so we back out R:R from edge and win_rate:
    # edge = p_win * R_win + (1 - p_win) * R_loss
    # Assume R_loss = -1.0, solve for R_win:
    # R_win = (edge + (1 - p_win)) / p_win

    if win_rate <= 0 or win_rate >= 1:
        raise ValueError(f"win_rate must be in (0, 1), got {win_rate}")

    r_loss = -1.0
    r_win = (edge_per_trade - (1 - win_rate) * r_loss) / win_rate

    # Risk per trade in R units (normalized to max DD)
    # If account is $50k, max DD is $2k, and risk/trade is 0.5%, that's $250 per trade
    # In R units: $250 / $2000 = 0.125 R per trade
    risk_per_trade_r = (risk_per_trade_pct / 100.0) * rules.account_size / rules.max_drawdown

    # Target and DD in R units (relative to max DD)
    target_r = rules.profit_target / rules.max_drawdown
    max_dd_r = 1.0  # by definition

    # Simulate
    pass_count = 0
    fail_count = 0
    timeout_count = 0
    trade_counts = []

    for _ in range(n_sims):
        balance_r = 0.0  # start at 0R (account at starting balance)
        peak_r = 0.0
        n_trades = 0
        days_traded = 0
        current_day_traded = False

        # Simulate until pass, fail, or timeout
        max_trades = 10_000  # safety cap
        for trade_idx in range(max_trades):
            # Check time limit (assume 1 trade per bar, ~50 trades per day for intraday)
            if rules.time_limit_days is not None:
                approx_days = n_trades / 50.0
                if approx_days > rules.time_limit_days:
                    timeout_count += 1
                    trade_counts.append(n_trades)
                    break

            # Draw trade outcome
            outcome_r = r_win if rng.random() < win_rate else r_loss
            pnl_r = outcome_r * risk_per_trade_r
            balance_r += pnl_r
            n_trades += 1

            # Update peak for trailing DD
            if balance_r > peak_r:
                peak_r = balance_r

            # Check pass condition
            if balance_r >= target_r:
                pass_count += 1
                trade_counts.append(n_trades)
                break

            # Check fail condition
            if rules.trailing_dd:
                drawdown_from_peak = peak_r - balance_r
                if drawdown_from_peak >= max_dd_r:
                    fail_count += 1
                    trade_counts.append(n_trades)
                    break
            else:
                if balance_r <= -max_dd_r:
                    fail_count += 1
                    trade_counts.append(n_trades)
                    break
        else:
            # Hit max_trades without resolution — count as timeout
            timeout_count += 1
            trade_counts.append(n_trades)

    total = pass_count + fail_count + timeout_count
    return {
        "pass_count": pass_count,
        "fail_count": fail_count,
        "timeout_count": timeout_count,
        "pass_pct": 100.0 * pass_count / total,
        "fail_pct": 100.0 * fail_count / total,
        "timeout_pct": 100.0 * timeout_count / total,
        "median_trades": int(np.median(trade_counts)),
        "mean_trades": float(np.mean(trade_counts)),
        "edge_per_trade": edge_per_trade,
        "win_rate": win_rate,
        "r_win": r_win,
        "r_loss": r_loss,
        "risk_per_trade_pct": risk_per_trade_pct,
        "risk_per_trade_r": risk_per_trade_r,
    }


def edge_required_for_pass_rate(
    target_pass_pct: float,
    win_rate_start: float,
    risk_per_trade_pct: float,
    rules: CombineRules,
    n_sims: int = 10_000,
    seed: int = 0,
) -> float:
    """
    Binary search to find edge required to achieve target pass rate.

    Returns edge_per_trade that produces ~target_pass_pct.
    """
    edge_lo, edge_hi = -0.5, 1.0
    tolerance = 0.5  # within 0.5% of target

    for _ in range(20):  # max iterations
        edge_mid = (edge_lo + edge_hi) / 2.0

        # Compute implied win rate from edge (assuming 1.5:1 R:R)
        # edge = p * 1.5 + (1-p) * (-1) = 2.5p - 1
        # p = (edge + 1) / 2.5
        win_rate = (edge_mid + 1.0) / 2.5
        win_rate = np.clip(win_rate, 0.01, 0.99)

        result = simulate_combine(edge_mid, win_rate, risk_per_trade_pct, rules, n_sims, seed)
        pass_pct = result["pass_pct"]

        if abs(pass_pct - target_pass_pct) < tolerance:
            return edge_mid

        if pass_pct < target_pass_pct:
            edge_lo = edge_mid
        else:
            edge_hi = edge_mid

    return edge_mid


if __name__ == "__main__":
    # Example: $50k account, $3k target, $2k max DD, trailing
    rules = CombineRules(
        account_size=50_000,
        profit_target=3_000,
        max_drawdown=2_000,
        trailing_dd=True,
    )

    print("Example: $50k account, $3k target, $2k trailing DD")
    print("=" * 60)

    # Zero edge (coin flip)
    result = simulate_combine(
        edge_per_trade=0.0,
        win_rate=0.40,
        risk_per_trade_pct=0.5,
        rules=rules,
        n_sims=10_000,
        seed=0
    )
    print(f"\nZero edge (coin flip at 1.5:1):")
    print(f"  Win rate: {result['win_rate']:.1%}")
    print(f"  Pass: {result['pass_pct']:.1f}%  Fail: {result['fail_pct']:.1f}%")
    print(f"  Median trades to resolution: {result['median_trades']}")

    # Modest edge
    result = simulate_combine(
        edge_per_trade=0.10,
        win_rate=0.44,
        risk_per_trade_pct=0.5,
        rules=rules,
        n_sims=10_000,
        seed=0
    )
    print(f"\n+0.10R edge:")
    print(f"  Win rate: {result['win_rate']:.1%}")
    print(f"  Pass: {result['pass_pct']:.1f}%  Fail: {result['fail_pct']:.1f}%")
    print(f"  Median trades to resolution: {result['median_trades']}")

    # Find edge for 50% pass rate
    edge_50 = edge_required_for_pass_rate(
        target_pass_pct=50.0,
        win_rate_start=0.44,
        risk_per_trade_pct=0.5,
        rules=rules,
        n_sims=10_000,
        seed=0
    )
    win_50 = (edge_50 + 1.0) / 2.5
    print(f"\nEdge required for 50% pass rate:")
    print(f"  Edge: +{edge_50:.3f}R/trade")
    print(f"  Win rate: {win_50:.1%} (at 1.5:1 R:R)")
