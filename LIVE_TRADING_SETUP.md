# Live Trading System - Setup Complete ✅

## 🎯 System Overview

Your automated ORB trading system is now fully configured with:

### ✅ Key Features Implemented

1. **Correlation-Aware Risk Scaling**
   - Base risk: 1.0% ($100)
   - With 1 correlated position: $75 (25% reduction)
   - With 2 correlated positions: $50 (50% reduction)
   - All NAS100 positions are considered correlated

2. **Smart Contract Selection**
   - **NAS100**: 100 bundles/lot @ $8 commission
   - **NAS100.mini**: 1 bundle/lot @ $8 commission
   - System automatically chooses NAS100 unless granularity requires mini
   - Avoids minis when possible (same $8 commission for 1/100th the size)

3. **Regime-Adaptive Trailing Stops**
   - Low volatility (ATR < 120): 0.75× multiplier
   - High volatility (ATR ≥ 120): 1.0× multiplier
   - Breakeven: 0.2R (move stop to entry)
   - Trail start: 0.5R (begin trailing)

4. **Comprehensive Risk Controls**
   - Max notional exposure: $250,000 (25× leverage)
   - Max daily loss: $500
   - Max total drawdown: $1,000
   - Position tracking with exposure limits

---

## 📋 Configuration Summary

```
Account: ALPHA#D#2386007 (Demo)
Server: ALPHA
Account Size: $10,000
Max Leverage: 25× ($250,000 notional)

Trading Hours: 13:30 - 20:00 UTC (NY Session)
Instrument: NAS100 / NAS100.mini
```

---

## 🚀 Running the System

### Step 1: Verify Configuration
```bash
python src/config_loader.py
```

Should show:
- ✅ Configuration loaded successfully
- Risk scaling with correlation (100 → 75 → 50)
- Contract specs (100 bundles vs 1 bundle)
- Optimal contract selection logic

### Step 2: Start Trading Engine (DEMO ONLY!)
```bash
python src/live_trading_engine.py
```

**⚠️ CRITICAL: Always start with DEMO account!**

---

## 🔍 What the Engine Does

### Startup
1. Authenticates with TradeLocker
2. Validates account balance and limits
3. Displays risk parameters and contract specs
4. Enters main monitoring loop

### During Session
1. **Opening Range (First 5 minutes)**
   - Tracks session high/low
   - Establishes range for breakout detection

2. **Entry Scanning**
   - Detects breakouts above/below range
   - Counts correlated positions
   - Scales risk accordingly (100 → 75 → 50)
   - Chooses optimal contract (NAS100 vs mini)
   - Checks exposure limits before entry

3. **Position Management**
   - Monitors profit in R multiples
   - Moves stop to breakeven at +0.2R
   - Activates trailing at +0.5R
   - Updates trailing stop every minute
   - Tracks closed positions

4. **Risk Monitoring**
   - Enforces daily loss limit ($500)
   - Enforces total drawdown limit ($1,000)
   - Prevents over-exposure (max $250k notional)

### Shutdown (Ctrl+C)
1. Closes all open positions
2. Cancels pending orders
3. Logs final state
4. Exits gracefully

---

## 📊 Position Sizing Examples

### Example 1: First Trade (No Correlated Positions)
```
ATR: 60 points (low vol)
ATR Multiplier: 0.75×
Stop Distance: 45 points
Risk Amount: $100 (base 1.0%)
Price: 20,000

Lot Size = $100 / 45 = 2.22 points worth
Per-point value = lot_size × multiplier = lot × 100

Using NAS100:
  0.022 lots × 100 bundles = 2.22 points
  Commission: $8
  Notional: $44,000
```

### Example 2: Second Trade (1 Correlated Position)
```
Same setup, but:
Risk Amount: $75 (reduced by 25%)

Lot Size = $75 / 45 = 1.67 points worth

Using NAS100:
  0.0167 lots × 100 bundles = 1.67 points
  Commission: $8
  Notional: $33,400
```

### Example 3: Third Trade (2 Correlated Positions)
```
Same setup, but:
Risk Amount: $50 (reduced by 50%)

Lot Size = $50 / 45 = 1.11 points worth

Using NAS100:
  0.0111 lots × 100 bundles = 1.11 points
  Commission: $8
  Notional: $22,200
```

**Total Exposure**: $44k + $33.4k + $22.2k = **$99.6k** (well under $250k limit)

---

## 🛡️ Safety Features

### Exposure Limits
- Checks total notional before each trade
- Rejects trades that would exceed 90% of max ($225k)
- Prevents over-leveraging during volatile periods

### Correlation Scaling
- Reduces risk as correlated positions accumulate
- Prevents portfolio concentration
- Maintains diversification across time

### Regime Adaptation
- Widens stops in high volatility (ATR ≥ 120)
- Tightens stops in low volatility (ATR < 120)
- Based on full-span backtest findings

### Smart Contract Choice
```python
# Prefers NAS100 over mini unless forced by granularity
if lot_size_nas100 >= 0.01:  # Minimum tradeable size
    return NAS100
else:
    return NAS100.mini  # Only when necessary
```

---

## 📈 Expected Performance (Based on Backtest)

**Full-Span Results (2019-2026):**
- Sharpe: 1.116
- Mean: +0.074R per trade
- Win Rate: 12.5%
- Consistency: 100% (positive in all 8 years)
- Total: +191R across 2,576 trades

**With $100 base risk:**
- Average trade: +$7.40
- 2,576 trades → +$19,062 (191% gain)
- Over 7.5 years = 25% annual return

**Note**: Past performance does not guarantee future results. Always test on demo first!

---

## 🚨 Pre-Launch Checklist

Before going live:

- [ ] Run on demo account for minimum 2 weeks
- [ ] Verify entry signals match backtest expectations
- [ ] Confirm trailing stops work correctly
- [ ] Monitor during high volatility periods
- [ ] Test regime switching (low → high ATR)
- [ ] Verify correlation scaling reduces risk properly
- [ ] Check commission costs match expectations
- [ ] Confirm max exposure limits work
- [ ] Test graceful shutdown (Ctrl+C)
- [ ] Review logs for any errors

---

## 📁 File Structure

```
orb-backtest/
├── .env                          # Your credentials (NEVER commit!)
├── .env.template                 # Template for credentials
├── src/
│   ├── config_loader.py          # Configuration + correlation logic
│   ├── tradelocker_client.py     # TradeLocker API wrapper
│   ├── position_manager.py       # Position tracking + correlation
│   └── live_trading_engine.py    # Main trading loop
├── logs/
│   └── trading.log               # Runtime logs (auto-created)
└── LIVE_TRADING_SETUP.md         # This file
```

---

## 🐛 Troubleshooting

### "Authentication failed"
- Check .env credentials
- Verify account is active
- Confirm server is "ALPHA"

### "Position size exceeds max"
- ATR stop too tight for available risk
- System will skip trade (logged)

### "Trade would exceed max exposure"
- Already at $225k+ notional
- Wait for positions to close

### "Outside trading hours"
- System only trades 9:30 AM - 4:00 PM ET (13:30-20:00 UTC)
- Will resume automatically when session opens

---

## 📞 Next Steps

1. **Test configuration**: `python src/config_loader.py`
2. **Start demo trading**: `python src/live_trading_engine.py`
3. **Monitor logs**: `tail -f logs/trading.log` (or open in text editor)
4. **Review daily**: Check trades, risk scaling, exposure
5. **After 2 weeks**: Decide if ready for live account

**Remember**: Start small, monitor closely, and only trade what you can afford to lose!

---

## 🎓 Key Learnings from Backtest

1. **ATR 0.75× beats 1.0×** (Sharpe 1.116 vs 1.059)
2. **Aggressive breakeven works** (0.2R better than 0.25R or 0.3R)
3. **Trail start timing doesn't matter** (0.4R, 0.5R, 0.6R all similar)
4. **Perfect consistency** (100% of years positive across 8 years)
5. **Regime adaptation optional** (but adds robustness during volatility spikes)

These findings are now embedded in your live trading system!

---

**Good luck, and trade safely! 🚀**
