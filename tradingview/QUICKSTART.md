# TradingView Quick Reference Card

## ⚡ 30-Second Setup

1. **Chart**: NAS100, 5-minute timeframe
2. **Pine Editor**: Paste `orb_strategy.pine` → Add to Chart
3. **Settings → Properties**:
   - Initial Capital: `$10,000`
   - Commission: `$8` per contract
   - Recalculate: "On bar close"
4. **Settings → Inputs**:
   - Entry Mode: `CC`
   - Enable Regime-Adaptive ATR: ✅
   - ATR Threshold: `115.0`
   - Low Vol Multiplier: `0.75`
   - High Vol Multiplier: `1.0`

---

## 📊 Expected Results (2019-2026)

| Metric | Target |
|--------|--------|
| Total Trades | ~2,500 |
| Sharpe Ratio | 1.0-1.2 |
| Win Rate | 10-15% |
| Avg Win/Loss | 6-10× |
| Max Drawdown | 8-12% |
| Total Return | +180-200 R |

---

## 🎨 Visual Guide

### Colors
- 🟦 **Blue shading** = Opening range forming (9:30-9:35 AM)
- 🟩 **Green line** = OR High (long breakout level)
- 🟥 **Red line** = OR Low (short breakdown level)
- 🟦 **Teal background** = Low volatility (ATR < 115)
- 🟧 **Orange background** = High volatility (ATR ≥ 115)

### Markers
- ✕ **Red cross** = Current stop level
- ✕ **Green cross** = Target level (1.0 R)
- 📍 **Label** = Regime indicator with ATR value

---

## 🔄 Paper Trading Workflow

1. **Enable Bar Replay**: Click ▶️ (Alt+R)
2. **Select start date**: 3-6 months ago
3. **Set speed**: 100× (1 day = ~5 minutes)
4. **Watch trades execute** automatically
5. **Review results** in Strategy Tester

---

## 🚨 Entry Signals (CC Mode)

### Long Entry
```
Close[0] > OR High AND Close[-1] > OR High
(Two consecutive 5-min closes above opening range)
```

### Short Entry
```
Close[0] < OR Low AND Close[-1] < OR Low
(Two consecutive 5-min closes below opening range)
```

---

## 🎯 Exit Rules

| Exit Type | Trigger | Action |
|-----------|---------|--------|
| **Target** | +1.0 R profit | Close at target price |
| **Breakeven** | +0.2 R profit | Move stop to entry, cancel target |
| **Trailing** | +0.5 R profit | Start trailing at ATR × multiplier |
| **Stop** | Hit stop level | Close at stop |
| **EOD** | 4:00 PM ET | Force close all |

---

## 🧪 Quick Tests

### Test 1: Low Vol Period
- Date: **Jan 2024**
- Expected: Teal background, tight stops, ~15-20 trades/month

### Test 2: High Vol Period
- Date: **June 2022**
- Expected: Orange background, wide stops, ~10-15 trades/month

### Test 3: Entry Mode Comparison
Run same period with:
- **CC** (default): Balanced
- **R-CC**: More trades, lower quality
- **II**: Fewer trades, higher quality

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| **0 trades** | Chart must be 5-min, session start = 9:30 |
| **Too many trades** | Properties → Recalculate "On bar close" |
| **Stops don't trail** | ATR Period must be 48 |
| **Results differ** | Verify Entry Mode = CC, ATR Threshold = 115 |

---

## 📈 Validation Checklist

Before going live:

- [ ] ✅ Paper traded 3+ months without errors
- [ ] ✅ Sharpe ratio 1.0-1.2
- [ ] ✅ Max drawdown <12%
- [ ] ✅ Results within 10% of Python backtest
- [ ] ✅ Regime switching works correctly
- [ ] ✅ Alerts delivered reliably
- [ ] ✅ Documented all edge cases

---

## 🎓 Key Parameters (Optimized 2019-2026)

| Parameter | Value | Why |
|-----------|-------|-----|
| ATR Low Vol | 0.75× | Best Sharpe (1.116) |
| ATR High Vol | 1.0× | Handles volatility spikes |
| Breakeven | 0.2 R | Protects capital early |
| Trail Start | 0.5 R | Captures runners |
| Target | 1.0 R | Risk:reward baseline |
| Threshold | 115 pts | Separates regimes cleanly |

---

**Full guide**: See `tradingview/README.md`  
**Strategy code**: `tradingview/orb_strategy.pine`  
**Backtest results**: `outputs/orb_full_span_exploration.json`
