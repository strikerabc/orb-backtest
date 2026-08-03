# ORB Backtest Engine — NQ & ES Futures

Research-grade Opening Range Breakout backtest for NQ and ES E-mini futures.
Exhaustive parameter sweep across all entry variants with rich per-trade journaling.

## What it does

- **1,944 variants** across 2 instruments × 3 sessions × 27 entry-mode/closure-tf combos × 2 directions × 6 RR levels
- **7 years of CME Globex data** (2019–2026, Databento GLBX.MDP3 ohlcv-1m)
- **10 non-overlapping regime windows** with even temporal spread across bull/bear/volatile/quiet markets
- **Per-trade journal** with 40+ fields including MAE/MFE, ATR cap flag, same-bar ambiguity flag, opposite-boundary flag, and volatility enrichment
- **Null calibration**: random-entry bootstrap benchmark so you can distinguish edge from luck
- **Block-bootstrap 95% CI** on per-variant expectancy

## Sessions

| Session | Local open | Local exit | DST schedule |
|---------|-----------|-----------|--------------|
| NY | 09:30 America/New_York | 12:00 (11:59 bar) | 2nd Sun Mar → 1st Sun Nov |
| London | 08:00 Europe/London | 12:00 (11:59 bar) | Last Sun Mar → last Sun Oct |
| Tokyo | 09:00 Asia/Tokyo | 12:00 (11:59 bar) | No DST — fixed JST UTC+9 |

## Entry modes (per range size)

| Mode | Description |
|------|-------------|
| II | Immediate Identification — first bar touching boundary+1 tick |
| CC | Candle Close — first close beyond boundary (1/5/15/30m closure TF) |
| TI | Tap-in — re-touch of boundary after initial breakout |
| R-II | Retest-II — II trigger after a prior tap-in |
| R-CC | Retest-CC — candle close beyond boundary after a prior tap-in |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Databento API key (first run only — builds the data cache)
export DATABENTO_API_KEY=db-xxxxxxxxxxxx   # Windows: $env:DATABENTO_API_KEY="db-..."

# 3. Run the full sweep
python main.py

# 4. Outputs in outputs/
#    trade_log.parquet / trade_log.csv   — ~1M rows, all trade details
#    summary.parquet / summary.csv       — per-variant aggregated metrics
#    regime_summary.parquet / .csv       — per-variant × regime-window metrics
#    report.md                           — top/bottom variants + regime stability
```

## Cost model (configurable in src/config.py)

```python
COMMISSION_PER_SIDE_USD    = 2.50   # per contract per side
SLIPPAGE_TICKS_ROUND_TRIP  = 1      # total round-trip ticks
```
Both `gross_r` and `net_r` are logged on every trade row.
Set both to 0 for frictionless gross-only analysis.

## Key design decisions

- **Swing SL (A5):** searches for down-closing clusters only *before* the initial breakout bar — slow retrace clusters above entry are excluded by construction
- **Exit bar (A10):** the 11:59 local bar (close = exactly 12:00:00 local)
- **ATR cap:** trades where TP > 2.5 × 4h ATR are flagged `atr_exceeds_cap=True` and simulated anyway — apply the filter at analysis time
- **Same-bar ambiguity:** if SL and TP are both hit in one 1m bar, SL is assumed first (conservative); flagged in `same_bar_ambiguous`
- **Data:** genuine CME Globex OHLCV (Databento GLBX.MDP3 continuous contracts, back-adjusted); cached locally in `data/`

## Project structure

```
src/
  config.py          all parameters
  data_layer.py      Databento download + local parquet cache
  range_builder.py   session-day extraction, opening ranges
  swing_detector.py  pre-breakout down-close cluster SL
  entry_detector.py  II/CC/TI/R-II/R-CC vectorised detection
  trade_sim.py       exit walk, MAE/MFE, cost model
  journal.py         per-trade row assembly
  regime_sampler.py  10 non-overlapping windows
  null_calibrator.py random-entry benchmark + bootstrap CI
  stats.py           per-variant metrics + bootstrap
  report.py          output writer
tests/
  test_swing_detector.py
  test_entry_sm.py
  test_sim.py
main.py              orchestration
```
