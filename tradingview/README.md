# TradingView Paper Trading Setup Guide

This guide shows how to use the ORB ATR Trailing Stop strategy in TradingView for automated paper trading validation.

---

## 🎯 Quick Start (5 Minutes)

### Step 1: Add the Strategy to TradingView

1. Open TradingView: https://www.tradingview.com/chart/
2. Search for **NAS100** (or **NQ1!** for futures)
3. Click **Pine Editor** at the bottom of the screen
4. Copy the entire contents of `orb_strategy.pine`
5. Paste into the Pine Editor
6. Click **"Add to Chart"** (or press Ctrl+S)

### Step 2: Configure the Strategy

The strategy will appear in the left panel. Click the ⚙️ gear icon to open settings:

#### **Properties Tab**:
- **Initial Capital**: $10,000 (matches your AlphaFunded account)
- **Order Size**: 0.02 lots (= 2 NAS100 contracts, $2/point)
- **Commission**: $8 per contract (round trip)
- **Slippage**: 2 ticks
- **Recalculate**: "On bar close" (prevents repainting)

#### **Inputs Tab** (Strategy Parameters):

**Session Settings**:
- Session Start Hour: `9` (9:30 AM ET)
- Session Start Minute: `30`
- Opening Range Duration: `5` minutes

**Entry Mode**:
- Entry Mode: `CC` (our best-performing variant)

**ATR Trailing Stop**:
- Enable Regime-Adaptive ATR: ✅ **Checked**
- ATR Regime Threshold: `115.0` points
- ATR Multiplier (Low Vol): `0.75`
- ATR Multiplier (High Vol): `1.0`
- ATR Period: `48` (= 4 hours on 5-min chart)

**Risk Management**:
- Breakeven at R: `0.2`
- Start Trailing at R: `0.5`
- Base R Target: `1.0`

**Position Sizing**:
- Risk Per Trade: `1.0%`
- Max Daily Loss: `5.0%`
- Max Total Drawdown: `8.0%`

### Step 3: Select Chart Timeframe

**Important**: The strategy is designed for **5-minute bars**.

- Set chart timeframe to **5m**
- The strategy will calculate the 4-hour ATR from the 5-minute data (48 periods = 4 hours)

### Step 4: View Strategy Results

Once applied, you'll see:

1. **Opening Range boxes** (light blue shading during first 5 mins)
2. **OR High/Low lines** (green/red horizontal lines)
3. **Current stop levels** (red crosses when in position)
4. **Target levels** (green crosses when target is active)
5. **Regime indicator** (label showing "LOW VOL" or "HIGH VOL" with current ATR)
6. **Background color**: Teal = low vol, Orange = high vol

**Strategy Tester** (bottom panel) shows:
- Net Profit
- Total Trades
- Win Rate
- Sharpe Ratio
- Max Drawdown
- Trade list with entry/exit prices

---

## 📊 Validating Against Python Backtest

### Expected Results (2019-2026 full span):

| Metric | Python Backtest | TradingView Target |
|--------|----------------|-------------------|
| **Total Trades** | 2,576 | ~2,500-2,600 |
| **Mean R** | +0.074 R | ~+0.07 R |
| **Sharpe Ratio** | 1.116 | ~1.0-1.2 |
| **Win Rate** | 12.5% | ~10-15% |
| **Total Return** | +191 R | ~+180-200 R |

### Why Slight Differences Are Normal:

1. **Data differences**: TradingView uses CFD data; Python used futures data
2. **Timing precision**: Pine Script bar-close vs our bar-open logic
3. **Session detection**: TradingView timezone vs UTC conversion
4. **Slippage model**: TradingView's simplified model vs reality

**If results are within 10% of Python backtest = ✅ validated**

---

## 🔄 Paper Trading with Replay Mode

TradingView's **Bar Replay** feature lets you paper trade through historical data at any speed:

### Step 1: Enable Bar Replay

1. Click the **play button** ▶️ in the top toolbar (or press Alt+R)
2. A timeline scrubber appears at the bottom
3. Select a start date (e.g., 3 months ago)

### Step 2: Configure Replay Speed

- **1x**: Real-time (1 minute = 1 minute)
- **10x**: 10× speed (6 minutes = 1 hour)
- **100x**: Fast-forward (36 seconds = 1 hour)
- **1000x**: Ultra-fast (3.6 seconds = 1 hour)

**Recommended for testing**: 100x speed = one trading day in ~5 minutes

### Step 3: Watch the Strategy Execute

As the replay advances:
- Opening range forms at 9:30 AM ET
- Entry signals trigger automatically
- Stops and targets adjust in real-time
- Regime indicator updates as ATR changes
- Trades close at stop, target, or EOD

### Step 4: Review Trade-by-Tade

Click on any trade marker to see:
- Entry price & time
- Exit price & time
- R result (profit/loss in R terms)
- Why it exited (stop/target/trail/EOD)

---

## 🎨 Visual Indicators Explained

### Opening Range
- **Light blue background**: Opening range forming (first 5 mins)
- **Green line**: OR High breakout level
- **Red line**: OR Low breakdown level

### Entry Signals
- **Green arrow ↑**: Long entry (close > OR High for 2 bars)
- **Red arrow ↓**: Short entry (close < OR Low for 2 bars)

### Trade Management
- **Red cross ✕**: Current stop level
- **Green cross ✕**: Target level (disappears after breakeven)
- Stop moves up/down as trailing activates

### Regime Indicator
- **Teal background + "LOW VOL"**: ATR < 115 → using 0.75× trail distance
- **Orange background + "HIGH VOL"**: ATR > 115 → using 1.0× trail distance
- Label shows current ATR value

---

## 🧪 Testing Different Scenarios

### Test 1: Low Volatility Period (2024 Q1)
- Set replay to January 2024
- ATR should stay ~80-100 (teal background)
- Tighter stops = fewer whipsaws

### Test 2: High Volatility Period (2022 Bear Market)
- Set replay to June 2022
- ATR should spike to 120-150 (orange background)
- Wider stops = longer hold times

### Test 3: Different Entry Modes
Change **Entry Mode** in settings:
- **R-CC**: Range Close Close (most aggressive)
- **CC**: Close Close (balanced, default)
- **II**: Wick Wick (most conservative)

Run the same date range with each mode to compare trade count and quality.

### Test 4: ATR Multiplier Sensitivity
Try these configs from our top-20 backtest:
- **0.75× constant**: Best Sharpe, tighter trails
- **1.0× constant**: Higher total R, wider trails
- **Regime-adaptive** (default): Best of both worlds

---

## 🚨 Alerts for Live Trading

Once validated with paper trading, set up alerts for live execution:

### Step 1: Create Alert

1. Right-click on chart → **"Add Alert"**
2. Condition: **Strategy alert → Any alert() function call**
3. Options:
   - Alert name: "ORB Signal"
   - Frequency: "Only Once"
   - Expiration: "Open-ended"

### Step 2: Alert Message Template

```
🎯 ORB Signal: {{ticker}}
Direction: {{strategy.order.action}}
Entry Price: {{close}}
Stop: {{strategy.order.stop}}
Target: {{strategy.order.limit}}
Regime: {{plot_0}}
Timestamp: {{timenow}}
```

### Step 3: Delivery Options

- **TradingView Mobile App**: Push notifications
- **Email**: Alert summary
- **Webhook**: Send to your live trading bot (advanced)

---

## 📈 Performance Dashboard

### Key Metrics to Watch

**Strategy Tester** (bottom panel) shows:

| Metric | Good | Warning |
|--------|------|---------|
| **Net Profit %** | +10-20% | <5% or >30% |
| **Sharpe Ratio** | 1.0-1.2 | <0.8 or >1.5 |
| **Max Drawdown** | 8-12% | >15% |
| **Profit Factor** | 1.3-1.6 | <1.2 |
| **Win Rate** | 10-15% | <8% or >20% |
| **Avg Win/Avg Loss** | 6-10× | <5× |

**Red flags**:
- Win rate >20% = likely repainting (check settings)
- Drawdown >15% = position sizing too aggressive
- <100 trades = insufficient sample size

---

## 🔧 Troubleshooting

### Problem: Strategy shows 0 trades

**Cause**: Chart timeframe incorrect or session times wrong

**Fix**:
1. Set chart to **5-minute bars**
2. Check **Session Start Hour** = `9` (not 14 or 16)
3. Verify **TradingView timezone** matches ET (Edit → Settings → Timezone)

### Problem: Results very different from Python backtest

**Cause**: Entry mode mismatch or ATR period wrong

**Fix**:
1. Verify **Entry Mode** = `CC` (not R-CC or II)
2. Check **ATR Period** = `48` (not 14 or 20)
3. Confirm **ATR Regime Threshold** = `115.0`

### Problem: Too many trades or unrealistic win rate

**Cause**: Strategy repainting (recalculating on every tick)

**Fix**:
1. Strategy Properties → **Recalculate** = "On bar close"
2. Disable **"Recalculate After Order Executed"**
3. Save and re-apply strategy

### Problem: Stops not trailing correctly

**Cause**: ATR too small or trail start too high

**Fix**:
1. Check ATR value (should be 80-150 for NAS100)
2. Verify **ATR Period** = `48` (not 14)
3. Lower **Start Trailing at R** to `0.4` for testing

---

## 🎓 Next Steps

### After Validation (1-3 months paper trading):

1. **Verify statistical match** to Python backtest (within 10%)
2. **Run replay through different regimes** (bull, bear, sideways)
3. **Test alert delivery** (make sure you receive signals)
4. **Document edge cases** (unusual OR sizes, gaps, holidays)

### Before Going Live:

1. ✅ 3 months minimum paper trading without errors
2. ✅ Sharpe >1.0, drawdown <10%
3. ✅ Alerts delivered reliably (<5 second delay)
4. ✅ Stop/target logic matches expectations in all scenarios
5. ✅ Regime switching works correctly across volatility shifts

### Going Live Checklist:

- [ ] AlphaFunded account funded
- [ ] TradeLocker credentials in `.env`
- [ ] Python live trading engine tested in demo mode
- [ ] Position sizing matches TradingView (0.02 lots = 2 contracts)
- [ ] Risk limits set (1% per trade, 5% daily, 8% total)
- [ ] Alerts connected to live execution system
- [ ] Emergency kill switch configured

---

## 📝 Strategy Logic Summary

For reference when debugging or explaining to others:

### Entry Logic (CC Mode)
```
OR High/Low = first 5 minutes of NY session (9:30-9:35 AM ET)

Long Signal:
- Close[0] > OR High AND
- Close[-1] > OR High
(two consecutive closes above opening range)

Short Signal:
- Close[0] < OR Low AND
- Close[-1] < OR Low
(two consecutive closes below opening range)

Entry: Close of signal bar
Initial Stop: Opposite side of OR (OR Low for longs, OR High for shorts)
```

### Exit Logic
```
1. Target Hit:
   - Exit at entry + 1.0R (for longs) or entry - 1.0R (for shorts)

2. Breakeven:
   - When profit reaches 0.2R, move stop to entry
   - Cancel target (now trailing only)

3. Trailing Stop:
   - Activates at 0.5R profit
   - Distance = ATR(48) × multiplier
   - Multiplier = 0.75× (low vol) or 1.0× (high vol)
   - Trails on every bar while profit increases

4. End of Day:
   - Force close any open position at 4:00 PM ET

5. Stop Hit:
   - Exit at current stop level (initial, breakeven, or trailing)
```

### Regime Detection
```
ATR(48) = Average True Range over 48 bars (= 4 hours on 5-min chart)

Low Volatility:  ATR < 115 → use 0.75× trail distance (tighter)
High Volatility: ATR ≥ 115 → use 1.0× trail distance (wider)

Adapts every bar as ATR changes
```

---

## 📚 Additional Resources

- **TradingView Pine Script Docs**: https://www.tradingview.com/pine-script-docs/
- **Strategy Tester Guide**: https://www.tradingview.com/support/solutions/43000481029/
- **Bar Replay Tutorial**: https://www.tradingview.com/support/solutions/43000628337/
- **Alert Setup Guide**: https://www.tradingview.com/support/solutions/43000520149/

---

## ⚠️ Disclaimer

This strategy is for **educational and backtesting purposes only**.

Past performance does not guarantee future results. Paper trading results may differ from live trading due to:
- Slippage and latency
- Order rejection and partial fills
- Market gaps and liquidity
- Data feed differences
- Psychological factors

Always test thoroughly in demo mode before risking real capital.

---

**Questions or issues?** Check the troubleshooting section or review the full backtest methodology in `orb-backtest/docs/METHODOLOGY.md`.
