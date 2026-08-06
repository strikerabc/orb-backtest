"""Equity curve for the best 6E setup, with 3-month recent window.

6E has zero profitable setups. The "best" is the least-bad: NY/30m/R-CC/15m/0.25RR
loses $44,332 over 7 years under corrected costs + hybrid sizing.

This script isolates that setup and renders two views:
  - Full history (2019-01 → 2026-01)
  - Last 3 months (2025-11-01 onward)
to show whether recent performance differs from the long-run average.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import OUTPUTS_DIR, INVALID_REASONS
from src.contracts import pnl_usd, size_hybrid

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
_OUT = Path(__file__).resolve().parent / OUTPUTS_DIR
BEST_6E = ("6E", "NY", 30, "R-CC", 15, 0.25)
LAST_3MO_START = "2025-11-01"


def load() -> pd.DataFrame:
    cols = ["instrument", "session", "range_minutes", "entry_mode", "closure_tf",
            "rr", "date", "entry_time_utc", "r_ticks", "gross_r", "tp_unfillable",
            "exit_reason", "tp_ticks"]
    df = pq.read_table(_OUT / "trade_log.parquet", columns=cols).to_pandas()
    df = df.loc[~df["tp_unfillable"].fillna(False).astype(bool)]
    df = df[~df["exit_reason"].isin(INVALID_REASONS)]
    df = df[df["gross_r"].notna()]
    sym, sess, rm, em, ct, rr = BEST_6E
    df = df[(df.instrument == sym) & (df.session == sess)
            & (df.range_minutes == rm) & (df.entry_mode == em)
            & (df.closure_tf == ct) & (df.rr == rr)].copy()
    log.info("loaded %d trades for %s/%s/%dm/%s/%dm/%.2fRR", len(df),
             sym, sess, rm, em, ct, rr)
    return df


def build_curve(df: pd.DataFrame) -> dict:
    d = df.sort_values(["date", "entry_time_utc"]).reset_index(drop=True)
    sz = size_hybrid(d["r_ticks"].to_numpy(), "6E")
    pnl = pnl_usd(d["gross_r"].to_numpy(), d["r_ticks"].to_numpy(), sz, "6E", "NY")
    taken = (sz["risk_usd"] > 0).to_numpy()
    pnl = np.where(taken, pnl, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        net_r = np.where(taken, pnl / np.maximum(sz["risk_usd"].to_numpy(), 1e-12), 0.0)
    d["cum_usd"] = np.cumsum(pnl)
    d["cum_r"] = np.nancumsum(net_r)
    d["date_ts"] = pd.to_datetime(d["date"])
    recent = d["date_ts"] >= pd.Timestamp(LAST_3MO_START)

    n = len(d)
    stats_all = {
        "period": "full",
        "date_start": str(d["date"].min()),
        "date_end": str(d["date"].max()),
        "trades": n,
        "win_rate": float((net_r[taken] > 0).mean()) if taken.any() else 0.0,
        "exp_net_r": float(net_r[taken].mean()) if taken.any() else 0.0,
        "total_usd": float(pnl.sum()),
        "commission_usd": float(sz.loc[taken, "commission_usd"].sum()),
        "mean_full": float(sz.loc[taken, "n_full"].mean()) if taken.any() else 0.0,
        "mean_micro": float(sz.loc[taken, "n_micro"].mean()) if taken.any() else 0.0,
    }

    rec = d[recent]
    pnl_rec = pnl[recent.to_numpy()]
    taken_rec = taken[recent.to_numpy()]
    net_r_rec = net_r[recent.to_numpy()]
    stats_recent = {
        "period": "last_3mo",
        "date_start": LAST_3MO_START,
        "date_end": str(d["date"].max()),
        "trades": int(recent.sum()),
        "win_rate": float((net_r_rec[taken_rec] > 0).mean()) if taken_rec.any() else 0.0,
        "exp_net_r": float(net_r_rec[taken_rec].mean()) if taken_rec.any() else 0.0,
        "total_usd": float(pnl_rec.sum()),
        "commission_usd": float(sz.loc[taken & recent, "commission_usd"].sum()),
        "mean_full": float(sz.loc[taken & recent, "n_full"].mean()) if (taken & recent).any() else 0.0,
        "mean_micro": float(sz.loc[taken & recent, "n_micro"].mean()) if (taken & recent).any() else 0.0,
    }

    # Downsample for charting
    k = max(1, n // 400)
    idx = list(range(0, n, k))
    if n and idx[-1] != n - 1:
        idx.append(n - 1)
    series = {
        "i": idx,
        "usd": [round(float(v), 2) for v in d["cum_usd"].to_numpy()[idx]],
        "r": [round(float(v), 4) for v in d["cum_r"].to_numpy()[idx]],
        "date": [str(v) for v in d["date"].to_numpy()[idx]],
    }

    recent_start_idx = int(recent.to_numpy().argmax()) if recent.any() else -1

    return {
        "stats_all": stats_all,
        "stats_recent": stats_recent,
        "series": series,
        "recent_start_idx": recent_start_idx,
    }


def render_html(data: dict) -> str:
    s_all = data["stats_all"]
    s_rec = data["stats_recent"]
    ser = data["series"]
    idx = data["recent_start_idx"]

    css = """
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',monospace;padding:24px}
h1{font-size:1.2rem;font-weight:600;color:#e6edf3;margin-bottom:8px}
.subtitle{font-size:.8rem;color:#8b949e;margin-bottom:16px}
.warn{background:#2d1a00;border:1px solid #9e6a03;border-radius:8px;padding:14px 18px;margin:16px 0;font-size:.82rem;color:#e3b341;line-height:1.6}
.warn strong{color:#f0b72f}
.chart-wrap{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:16px;margin:20px 0;height:340px}
.stats-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin:20px 0}
.stat-panel{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:16px}
.stat-title{font-size:.85rem;font-weight:600;color:#58a6ff;margin-bottom:12px;text-transform:uppercase;letter-spacing:.5px}
.stat-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1c2128}
.stat-row:last-child{border-bottom:none}
.stat-lbl{font-size:.75rem;color:#8b949e}
.stat-val{font-size:.8rem;font-weight:600;color:#e6edf3}
.red{color:#f85149}.green{color:#3fb950}.yellow{color:#d29922}
"""

    js = f"""
const data = {{
    labels: {ser['date']},
    datasets: [{{
        label: 'Cumulative P&L (USD)',
        data: {ser['usd']},
        borderColor: '#58a6ff',
        backgroundColor: '#58a6ff22',
        borderWidth: 1.8,
        pointRadius: 0,
        fill: true,
        tension: 0.05,
    }}]
}};
const recentIdx = {idx};
new Chart(document.getElementById('chart'), {{
    type: 'line',
    data: data,
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            legend: {{display: true, labels: {{color: '#8b949e', font: {{size: 10}}}}}},
            tooltip: {{
                callbacks: {{
                    title: (items) => items[0].label || '',
                    label: (c) => 'P&L: $' + c.parsed.y.toLocaleString(undefined, {{maximumFractionDigits: 0}})
                }}
            }},
            annotation: recentIdx >= 0 ? {{
                annotations: {{
                    line1: {{
                        type: 'line',
                        xMin: recentIdx,
                        xMax: recentIdx,
                        borderColor: '#d29922',
                        borderWidth: 2,
                        borderDash: [6, 3],
                        label: {{
                            display: true,
                            content: 'Last 3 months →',
                            position: 'start',
                            color: '#d29922',
                            font: {{size: 9}}
                        }}
                    }}
                }}
            }} : {{}}
        }},
        scales: {{
            x: {{ticks: {{color: '#6e7681', font: {{size: 8}}, maxTicksLimit: 8}}, grid: {{color: '#1c2128'}}}},
            y: {{ticks: {{color: '#58a6ff', font: {{size: 9}}, callback: (v) => '$' + (v/1000).toFixed(0) + 'k'}},
                grid: {{color: '#1c2128'}},
                title: {{display: true, text: 'USD', color: '#58a6ff', font: {{size: 9}}}}}}
        }}
    }}
}});
"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>6E Best Setup — 3-Month Recent vs Full History</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>{css}</style></head><body>
<h1>6E "Best" Setup — 3-Month Recent Window</h1>
<div class="subtitle">6E/NY/30m/R-CC/15m/0.25RR · Hybrid sizing · Corrected costs ($3.10 RT e-mini, $1.40 RT micro)</div>
<div class="warn">
<strong>Context: 6E has zero profitable setups.</strong> Every 6E configuration loses money over the full sample.
The setup shown here is the <strong>least-bad</strong> by total P&L — it loses $44,332 over 7 years at a $900 risk budget.
The chart compares <strong>recent performance (last 3 months)</strong> against the full historical record.
</div>
<div class="chart-wrap"><canvas id="chart"></canvas></div>
<div class="stats-grid">
<div class="stat-panel">
<div class="stat-title">Full History ({s_all['date_start']} → {s_all['date_end']})</div>
<div class="stat-row"><span class="stat-lbl">Trades</span><span class="stat-val">{s_all['trades']:,}</span></div>
<div class="stat-row"><span class="stat-lbl">Win rate</span><span class="stat-val yellow">{s_all['win_rate']:.2%}</span></div>
<div class="stat-row"><span class="stat-lbl">Exp/trade</span><span class="stat-val {'green' if s_all['exp_net_r']>0 else 'red'}">{s_all['exp_net_r']:+.4f}R</span></div>
<div class="stat-row"><span class="stat-lbl">Total P&L</span><span class="stat-val {'green' if s_all['total_usd']>0 else 'red'}">${s_all['total_usd']:+,.0f}</span></div>
<div class="stat-row"><span class="stat-lbl">Commission</span><span class="stat-val red">${s_all['commission_usd']:,.0f}</span></div>
<div class="stat-row"><span class="stat-lbl">Mean contracts</span><span class="stat-val">{s_all['mean_full']:.1f} full + {s_all['mean_micro']:.1f} micro</span></div>
</div>
<div class="stat-panel">
<div class="stat-title">Last 3 Months ({s_rec['date_start']} → {s_rec['date_end']})</div>
<div class="stat-row"><span class="stat-lbl">Trades</span><span class="stat-val">{s_rec['trades']:,}</span></div>
<div class="stat-row"><span class="stat-lbl">Win rate</span><span class="stat-val yellow">{s_rec['win_rate']:.2%}</span></div>
<div class="stat-row"><span class="stat-lbl">Exp/trade</span><span class="stat-val {'green' if s_rec['exp_net_r']>0 else 'red'}">{s_rec['exp_net_r']:+.4f}R</span></div>
<div class="stat-row"><span class="stat-lbl">Total P&L</span><span class="stat-val {'green' if s_rec['total_usd']>0 else 'red'}">${s_rec['total_usd']:+,.0f}</span></div>
<div class="stat-row"><span class="stat-lbl">Commission</span><span class="stat-val red">${s_rec['commission_usd']:,.0f}</span></div>
<div class="stat-row"><span class="stat-lbl">Mean contracts</span><span class="stat-val">{s_rec['mean_full']:.1f} full + {s_rec['mean_micro']:.1f} micro</span></div>
</div>
</div>
<script>{js}</script></body></html>"""


def main() -> None:
    df = load()
    data = build_curve(df)
    html = render_html(data)
    out_path = _OUT / "equity_6e_best_3mo.html"
    out_path.write_text(html, encoding="utf-8")
    log.info("\nFull history:   %d trades, $%+,.0f, %.2f%% win",
             data["stats_all"]["trades"], data["stats_all"]["total_usd"],
             100 * data["stats_all"]["win_rate"])
    log.info("Last 3 months:  %d trades, $%+,.0f, %.2f%% win",
             data["stats_recent"]["trades"], data["stats_recent"]["total_usd"],
             100 * data["stats_recent"]["win_rate"])
    log.info("\nwrote %s (%d bytes)", out_path, out_path.stat().st_size)
    return str(out_path)


if __name__ == "__main__":
    main()
