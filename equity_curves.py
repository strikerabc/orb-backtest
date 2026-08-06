"""Equity curves for the 18 requested setups, sized to a real account.

Each setup is `instrument / session / 15m range / CC entry / 5m closure /
long+short / rr`.  Long and short are merged into ONE chronological curve
because that is how they would be deployed -- both directions run every day.

Two curves per setup:

  * **Net R** -- the strategy's own scale, every trade weighted equally.
  * **Net USD** under the $900-risk / 10-contract cap, which is what the
    account actually experiences.  These two can disagree in shape: R assumes
    uniform risk per trade, while the contract cap makes narrow-stop trades
    under-deployed and wide-stop trades unaffordable outright.

In-sample runs 2019-01 -> 2026-01 (read from the sweep's trade log).  The
holdout period 2026-02-01 -> 2026-07 is re-simulated from cached 1m bars,
since the sweep's regime windows stop at 2026-01-31.  No API calls.
"""

from __future__ import annotations

import http.server
import json
import logging
import socketserver
import threading
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import INSTRUMENTS, OUTPUTS_DIR
from src.data_layer import _compute_enrichment, ensure_daily, ensure_data
from src.entry_detector import detect_entries
from src.range_builder import build_session_days
from src.sizing import MAX_CONTRACTS, MAX_RISK_USD, size_trades
from src.trade_sim import simulate_trade, slippage_ticks_for

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_OUT = _ROOT / OUTPUTS_DIR
HOLDOUT_START = "2026-02-01"
HTML_PATH = _OUT / "equity_curves_18.html"
PORT = 8765

# The 18 requested setups, in the order given.
RANGE_MIN, ENTRY_MODE, CLOSURE_TF = 15, "CC", 5
RRS = [0.25, 0.5, 0.75]
REQUESTED = [(sym, sess, rr)
             for sym in ("6E", "NQ")
             for sess in ("NY", "LDN", "TOK")
             for rr in RRS]


def available(sym: str, sess: str) -> bool:
    """6E/TOK is excluded in config: 09:00 JST is ~00:00 UTC, dead for EUR."""
    return sess in INSTRUMENTS[sym]["sessions"]


def load_in_sample() -> pd.DataFrame:
    """Read the requested setups out of the sweep's trade log."""
    cols = ["instrument", "session", "range_minutes", "entry_mode", "closure_tf",
            "direction", "rr", "date", "entry_time_utc", "r_ticks", "tp_ticks",
            "tp_unfillable", "net_r", "gross_r", "exit_reason"]
    log.info("reading in-sample trade log ...")
    df = pq.read_table(_OUT / "trade_log.parquet", columns=cols).to_pandas()
    df = df[(df.instrument.isin(["6E", "NQ"]))
            & (df.range_minutes == RANGE_MIN)
            & (df.entry_mode == ENTRY_MODE)
            & (df.closure_tf == CLOSURE_TF)
            & (df.rr.isin(RRS))]
    # Unfillable TPs are not real fills (stats.py drops them too).
    df = df.loc[~df["tp_unfillable"].fillna(False).astype(bool)]
    df["phase"] = "in_sample"
    log.info("  %d in-sample trades across %d setups", len(df),
             df.groupby(["instrument", "session", "rr"]).ngroups)
    return df


def simulate_holdout() -> pd.DataFrame:
    """Re-simulate 2026-02 onward from cached 1m bars (no API calls)."""
    rows = []
    for sym in ("6E", "NQ"):
        log.info("re-simulating holdout for %s ...", sym)
        bars = _compute_enrichment(ensure_data(sym), ensure_daily(sym),
                                   tick_size=INSTRUMENTS[sym]["tick_size"])
        bars = bars[bars["timestamp"] >= pd.Timestamp(HOLDOUT_START, tz="UTC")]
        if bars.empty:
            log.info("  no holdout bars")
            continue
        log.info("  bars %d  %s -> %s", len(bars),
                 bars["timestamp"].min().date(), bars["timestamp"].max().date())
        for sess in ("NY", "LDN", "TOK"):
            if not available(sym, sess):
                continue
            for sd in build_session_days(bars, sym, sess):
                for es in detect_entries(sd):
                    if (es.range_minutes != RANGE_MIN or es.mode != ENTRY_MODE
                            or es.closure_tf != CLOSURE_TF):
                        continue
                    for tr in simulate_trade(es, sd, RRS):
                        if tr.exit_reason in ("INVALID", None):
                            continue
                        if getattr(tr, "tp_unfillable", False):
                            continue
                        rows.append({
                            "instrument": sym, "session": sess,
                            "range_minutes": es.range_minutes,
                            "entry_mode": es.mode, "closure_tf": es.closure_tf,
                            "direction": es.direction, "rr": tr.rr,
                            "date": str(sd.local_date),
                            "entry_time_utc": str(
                                sd.bar_timestamps[es.entry_bar_idx]),
                            "r_ticks": tr.r_ticks, "tp_ticks": tr.tp_ticks,
                            "tp_unfillable": False,
                            "net_r": tr.net_r, "gross_r": tr.gross_r,
                            "exit_reason": tr.exit_reason,
                            "phase": "holdout",
                        })
    out = pd.DataFrame(rows)
    log.info("holdout trades simulated: %d", len(out))
    return out


def _max_dd(equity: np.ndarray) -> float:
    """Max peak-to-trough drawdown of a cumulative curve (absolute units)."""
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def build_curve(df: pd.DataFrame, sym: str) -> dict:
    """One chronological long+short curve, in R and in sized dollars."""
    d = df.sort_values(["date", "entry_time_utc"]).reset_index(drop=True)
    d = size_trades(d, sym)

    d["cum_r"] = d["net_r"].cumsum()
    # Unexecutable trades contribute 0 P&L but still occupy their slot in the
    # sequence -- pnl_usd is already 0 for them via contracts == 0.
    d["cum_usd"] = d["pnl_usd"].cumsum()

    ex = d[d.executable]
    n = len(d)
    n_ex = len(ex)
    wins = d["net_r"] > 0
    hold = d["phase"] == "holdout"

    stats = {
        "trades": n,
        "trades_executable": n_ex,
        "executable_pct": (n_ex / n) if n else 0.0,
        "win_rate": float(wins.mean()) if n else 0.0,
        "exp_net_r": float(d["net_r"].mean()) if n else 0.0,
        "total_net_r": float(d["net_r"].sum()),
        "total_usd": float(d["pnl_usd"].sum()),
        "max_dd_r": _max_dd(d["cum_r"].to_numpy()),
        "max_dd_usd": _max_dd(d["cum_usd"].to_numpy()),
        "sl_med": float(d["r_ticks"].median()) if n else 0.0,
        "sl_p10": float(d["r_ticks"].quantile(0.10)) if n else 0.0,
        "sl_p90": float(d["r_ticks"].quantile(0.90)) if n else 0.0,
        "tp_med": float(d["tp_ticks"].median()) if n else 0.0,
        "risk_med_usd": float(d["risk_per_contract_usd"].median()) if n else 0.0,
        "mean_contracts": float(ex["contracts"].mean()) if n_ex else 0.0,
        "capital_eff": float(d["risk_deployed_usd"].mean() / MAX_RISK_USD) if n else 0.0,
        "friction_r": (slippage_ticks_for(sym, d["session"].iloc[0])
                       / float(d["r_ticks"].median())) if n else float("nan"),
        "n_holdout": int(hold.sum()),
        "holdout_net_r": float(d.loc[hold, "net_r"].sum()),
        "holdout_usd": float(d.loc[hold, "pnl_usd"].sum()),
        "date_min": str(d["date"].min()) if n else "",
        "date_max": str(d["date"].max()) if n else "",
        # Index at which the holdout begins, for the chart divider.
        "holdout_idx": int(hold.to_numpy().argmax()) if hold.any() else -1,
    }
    # Downsample the plotted series: 1,700+ points per chart x 15 charts is
    # needless payload.  Keep every k-th point plus always the final one so the
    # endpoint (and therefore the visible total) is exact.
    k = max(1, n // 400)
    idx = list(range(0, n, k))
    if n and idx[-1] != n - 1:
        idx.append(n - 1)
    series = {
        "i": idx,
        "r": [round(float(v), 4) for v in d["cum_r"].to_numpy()[idx]],
        "usd": [round(float(v), 2) for v in d["cum_usd"].to_numpy()[idx]],
        "date": [str(v) for v in d["date"].to_numpy()[idx]],
    }
    return {"stats": stats, "series": series}


def collect() -> list[dict]:
    """Build every requested setup, marking the unavailable ones explicitly."""
    ins = load_in_sample()
    hol = simulate_holdout()
    allt = pd.concat([ins, hol], ignore_index=True) if len(hol) else ins

    panels = []
    for sym, sess, rr in REQUESTED:
        label = (f"{sym}/{sess}/{RANGE_MIN}m/{ENTRY_MODE}/{CLOSURE_TF}m/"
                 f"Long+Short/{rr}RR")
        if not available(sym, sess):
            panels.append({
                "label": label, "instrument": sym, "session": sess, "rr": rr,
                "unavailable": True,
                "reason": ("Not in the instrument universe: config restricts 6E "
                           "to LDN and NY. 09:00 JST is ~00:00 UTC, a dead zone "
                           "for EUR liquidity, so there is no Tokyo open to "
                           "trade."),
            })
            continue
        sub = allt[(allt.instrument == sym) & (allt.session == sess)
                   & (allt.rr == rr)]
        if sub.empty:
            panels.append({"label": label, "instrument": sym, "session": sess,
                           "rr": rr, "unavailable": True,
                           "reason": "No trades generated."})
            continue
        built = build_curve(sub, sym)
        panels.append({"label": label, "instrument": sym, "session": sess,
                       "rr": rr, "unavailable": False, **built})
        s = built["stats"]
        log.info("  %-42s n=%-5d wr=%.3f  netR=%+8.2f  usd=%+11.2f  "
                 "sl=%.0ft  x%.1f",
                 label, s["trades"], s["win_rate"], s["total_net_r"],
                 s["total_usd"], s["sl_med"], s["mean_contracts"])
    return panels


def serve(path: Path, port: int = PORT) -> str:
    """Serve the outputs dir so VS Code's Simple Browser can load the page."""
    directory = str(path.parent)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def log_message(self, *a):  # silence per-request noise
            pass

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/{path.name}"


CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',monospace;padding:20px 24px}
h1{font-size:1.15rem;font-weight:600;color:#e6edf3;letter-spacing:.5px;margin-bottom:4px}
h2{font-size:.95rem;font-weight:600;color:#e6edf3;margin:26px 0 12px;padding-bottom:6px;border-bottom:1px solid #21262d}
.subtitle{font-size:.78rem;color:#8b949e;margin-bottom:6px}
.badge{display:inline-block;background:#161b22;border:1px solid #30363d;border-radius:4px;padding:2px 8px;font-size:.72rem;color:#58a6ff;margin-right:6px}
.warn{background:#2d1a00;border:1px solid #9e6a03;border-radius:8px;padding:12px 16px;margin:14px 0 20px;font-size:.8rem;color:#e3b341;line-height:1.55}
.warn strong{color:#f0b72f}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
@media(max-width:1400px){.grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.panel{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:14px}
.panel.dead{opacity:.62;border-style:dashed}
.panel.pos{border-color:#238636}
.panel-title{font-size:.78rem;font-weight:600;color:#e6edf3;margin-bottom:2px}
.panel-sub{font-size:.68rem;color:#8b949e;margin-bottom:10px}
.chart-wrap{position:relative;height:190px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:10px}
.stat{background:#0d1117;border-radius:5px;padding:7px 9px;border:1px solid #21262d}
.stat-lbl{font-size:.6rem;color:#8b949e;text-transform:uppercase;letter-spacing:.4px;margin-bottom:2px}
.stat-val{font-size:.86rem;font-weight:700;color:#e6edf3}
.green{color:#3fb950}.red{color:#f85149}.yellow{color:#d29922}.blue{color:#58a6ff}
.dead-msg{font-size:.72rem;color:#8b949e;line-height:1.5;padding:22px 8px;text-align:center}
.tbl-wrap{background:#161b22;border:1px solid #21262d;border-radius:10px;overflow-x:auto;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:.71rem;white-space:nowrap}
th{background:#0d1117;color:#8b949e;text-align:right;padding:7px 9px;border-bottom:1px solid #21262d;font-weight:500}
th:first-child,td:first-child{text-align:left}
td{padding:6px 9px;border-bottom:1px solid #1c2128;text-align:right}
tr:last-child td{border-bottom:none}
tr:hover td{background:#1c2128}
.note{font-size:.71rem;color:#8b949e;margin-top:10px;padding:10px 13px;background:#161b22;border-radius:6px;border:1px solid #21262d;line-height:1.6}
.gate-fail{color:#f85149}.gate-pass{color:#3fb950}
"""


def _cls(v: float) -> str:
    return "green" if v > 0 else ("red" if v < 0 else "yellow")


def _panel_html(p: dict, i: int) -> str:
    if p["unavailable"]:
        return (f'<div class="panel dead"><div class="panel-title">{p["label"]}'
                f'</div><div class="panel-sub">unavailable</div>'
                f'<div class="dead-msg">{p["reason"]}</div></div>')
    s = p["stats"]
    pos = " pos" if s["total_usd"] > 0 else ""
    return f"""<div class="panel{pos}">
<div class="panel-title">{p["label"]}</div>
<div class="panel-sub">{s["trades"]:,} trades · {s["date_min"]} → {s["date_max"]}
  · SL {s["sl_med"]:.0f}t (p10 {s["sl_p10"]:.0f} / p90 {s["sl_p90"]:.0f})
  · TP {s["tp_med"]:.2f}t</div>
<div class="chart-wrap"><canvas id="c{i}"></canvas></div>
<div class="stats">
  <div class="stat"><div class="stat-lbl">Win rate</div>
    <div class="stat-val blue">{s["win_rate"]:.1%}</div></div>
  <div class="stat"><div class="stat-lbl">Net R</div>
    <div class="stat-val {_cls(s["total_net_r"])}">{s["total_net_r"]:+,.1f}</div></div>
  <div class="stat"><div class="stat-lbl">Net USD</div>
    <div class="stat-val {_cls(s["total_usd"])}">${s["total_usd"]:+,.0f}</div></div>
  <div class="stat"><div class="stat-lbl">Exp/trade</div>
    <div class="stat-val {_cls(s["exp_net_r"])}">{s["exp_net_r"]:+.4f}R</div></div>
  <div class="stat"><div class="stat-lbl">Max DD</div>
    <div class="stat-val red">${s["max_dd_usd"]:,.0f}</div></div>
  <div class="stat"><div class="stat-lbl">Contracts</div>
    <div class="stat-val yellow">{s["mean_contracts"]:.1f}×</div></div>
  <div class="stat"><div class="stat-lbl">Risk/contract</div>
    <div class="stat-val">${s["risk_med_usd"]:,.0f}</div></div>
  <div class="stat"><div class="stat-lbl">Friction</div>
    <div class="stat-val {"red" if s["friction_r"] > 0.15 else "green"}">{s["friction_r"]:.1%}R</div></div>
  <div class="stat"><div class="stat-lbl">Affordable</div>
    <div class="stat-val {"red" if s["executable_pct"] < 0.8 else "green"}">{s["executable_pct"]:.0%}</div></div>
</div></div>"""


def _table_html(panels: list[dict]) -> str:
    head = ("Setup|Trades|Win rate|Exp/trade|Net R|Net USD|Max DD $|SL med|TP med"
            "|Friction|Risk/ct|Contracts|Cap eff|Afford|Holdout R|Holdout $"
            ).split("|")
    rows = []
    for p in panels:
        if p["unavailable"]:
            rows.append(f'<tr><td>{p["label"]}</td>'
                        f'<td colspan="15" style="text-align:left;color:#8b949e">'
                        f'— not in universe (6E has no Tokyo session) —</td></tr>')
            continue
        s = p["stats"]
        rows.append(
            f'<tr><td>{p["label"]}</td>'
            f'<td>{s["trades"]:,}</td>'
            f'<td class="blue">{s["win_rate"]:.2%}</td>'
            f'<td class="{_cls(s["exp_net_r"])}">{s["exp_net_r"]:+.4f}</td>'
            f'<td class="{_cls(s["total_net_r"])}">{s["total_net_r"]:+,.2f}</td>'
            f'<td class="{_cls(s["total_usd"])}">${s["total_usd"]:+,.0f}</td>'
            f'<td class="red">${s["max_dd_usd"]:,.0f}</td>'
            f'<td>{s["sl_med"]:.0f}t</td>'
            f'<td>{s["tp_med"]:.2f}t</td>'
            f'<td class="{"gate-fail" if s["friction_r"] > 0.15 else "gate-pass"}">'
            f'{s["friction_r"]:.2%}</td>'
            f'<td>${s["risk_med_usd"]:,.0f}</td>'
            f'<td>{s["mean_contracts"]:.2f}×</td>'
            f'<td>{s["capital_eff"]:.0%}</td>'
            f'<td class="{"gate-fail" if s["executable_pct"] < 0.8 else "gate-pass"}">'
            f'{s["executable_pct"]:.0%}</td>'
            f'<td class="{_cls(s["holdout_net_r"])}">{s["holdout_net_r"]:+.2f}</td>'
            f'<td class="{_cls(s["holdout_usd"])}">${s["holdout_usd"]:+,.0f}</td>'
            f'</tr>')
    return ('<div class="tbl-wrap"><table><thead><tr>'
            + "".join(f"<th>{h}</th>" for h in head)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


JS = r"""
const OPTS = (hIdx) => ({
  responsive:true, maintainAspectRatio:false,
  interaction:{mode:'index',intersect:false},
  plugins:{
    legend:{display:true,labels:{color:'#8b949e',boxWidth:10,font:{size:9}}},
    tooltip:{callbacks:{title:(it)=>it[0].raw.d||'',
      label:(c)=>c.dataset.label+': '+(c.dataset.yAxisID==='yUsd'
        ? '$'+c.parsed.y.toLocaleString(undefined,{maximumFractionDigits:0})
        : c.parsed.y.toFixed(2)+'R')}},
  },
  scales:{
    x:{ticks:{color:'#6e7681',font:{size:8},maxTicksLimit:6},
       grid:{color:'#1c2128'}},
    yR:{position:'left',ticks:{color:'#58a6ff',font:{size:8}},
        grid:{color:'#1c2128'},title:{display:true,text:'R',color:'#58a6ff',
        font:{size:8}}},
    yUsd:{position:'right',ticks:{color:'#d29922',font:{size:8},
          callback:(v)=>'$'+(v/1000).toFixed(0)+'k'},
          grid:{drawOnChartArea:false},
          title:{display:true,text:'USD',color:'#d29922',font:{size:8}}},
  },
});

function draw(id, s, hIdx){
  const pts = s.i.map((v,j)=>({x:v, d:s.date[j]}));
  new Chart(document.getElementById(id), {
    type:'line',
    data:{labels:s.i.map((v,j)=>s.date[j]),
      datasets:[
        {label:'Cum net R', data:s.r.map((v,j)=>({x:j,y:v,d:s.date[j]})),
         yAxisID:'yR', borderColor:'#58a6ff', backgroundColor:'#58a6ff22',
         borderWidth:1.4, pointRadius:0, fill:true, tension:.05},
        {label:'Cum net USD ($900/10ct)',
         data:s.usd.map((v,j)=>({x:j,y:v,d:s.date[j]})),
         yAxisID:'yUsd', borderColor:'#d29922', borderWidth:1.4,
         pointRadius:0, fill:false, tension:.05, borderDash:[4,2]},
      ]},
    options:OPTS(hIdx),
  });
}
"""


def render(panels: list[dict]) -> str:
    live = [p for p in panels if not p["unavailable"]]
    n_pos = sum(1 for p in live if p["stats"]["total_usd"] > 0)
    tot_usd = sum(p["stats"]["total_usd"] for p in live)
    draws = "\n".join(
        f'draw("c{i}", {json.dumps(p["series"])}, {p["stats"]["holdout_idx"]});'
        for i, p in enumerate(panels) if not p["unavailable"])
    grid = "\n".join(_panel_html(p, i) for i, p in enumerate(panels))
    dmin = min((p["stats"]["date_min"] for p in live), default="")
    dmax = max((p["stats"]["date_max"] for p in live), default="")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ORB Equity Curves — 18 requested setups</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>{CSS}</style></head><body>
<h1>ORB Equity Curves — 18 Requested Setups</h1>
<div class="subtitle">
  <span class="badge">15m Range</span>
  <span class="badge">CC · 5m Close</span>
  <span class="badge">Long + Short merged</span>
  <span class="badge">{dmin} → {dmax}</span>
  <span class="badge">$900 risk · 10 contract cap</span>
</div>
<div class="warn">
  <strong>{len(live)} of 18 setups exist.</strong> The three 6E/TOK requests have
  no data: config restricts 6E to LDN and NY because 09:00 JST ≈ 00:00 UTC, a
  dead zone for EUR liquidity — there is no Tokyo open to trade.<br>
  <strong>{n_pos} of {len(live)} plotted setups end net-positive in dollars</strong>
  (combined {"+" if tot_usd >= 0 else ""}${tot_usd:,.0f} across all {len(live)}).
  Curves are in-sample through 2026-01-30, then re-simulated on holdout data
  (2026-02-01 →), which was never used for selection.<br>
  <strong>Two lines per panel:</strong> blue = cumulative net R (every trade
  weighted equally); amber dashed = cumulative dollars under your actual
  constraint. They differ because the contract cap under-deploys narrow-stop
  trades and makes wide-stop trades unaffordable outright.
</div>
<h2>Equity curves</h2>
<div class="grid">{grid}</div>
<h2>Summary — all 18 requested setups</h2>
{_table_html(panels)}
<div class="note">
  <strong>Friction</strong> = round-trip measured spread ÷ median stop, i.e. the
  share of each risk unit paid to the bid-ask. <strong>Risk/ct</strong> = median
  stop × tick value; <strong>Contracts</strong> = mean integer contracts at
  $900 (⌊900/risk⌋, capped at 10). <strong>Cap eff</strong> = mean dollars
  actually at risk ÷ $900 — below 100% means the 10-contract cap bound before
  the budget did. <strong>Afford</strong> = share of trades where even one
  contract fits inside $900; NQ/NY fails this because ORB swing-low stops there
  run past the 180-tick (45.00 point) wall that $900 imposes on full-size NQ.
</div>
<script>{JS}
{draws}
</script></body></html>"""


def main() -> None:
    panels = collect()
    HTML_PATH.write_text(render(panels), encoding="utf-8")
    log.info("\nwrote %s", HTML_PATH)
    url = serve(HTML_PATH)
    log.info("serving at %s", url)
    log.info("VS Code: Ctrl+Shift+P -> 'Simple Browser: Show' -> paste the URL")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    input("\npress Enter to stop the server ...\n")


if __name__ == "__main__":
    main()
