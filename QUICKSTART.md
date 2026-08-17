# Quick Start Guide: AlphaFunded Live Trading Setup

## 1. Get Your TradeLocker Credentials

### From AlphaFunded Dashboard:
1. Log into your AlphaFunded account at https://alphafunded.com
2. Navigate to your **Dashboard** → **Account Details**
3. Find your **TradeLocker credentials**:
   - Server: (e.g., `live.tradelocker.com` or `demo.tradelocker.com`)
   - Account ID: Your trading account number
   - Username: Your TradeLocker email
   - Password: Your TradeLocker password

## 2. Configure Your Environment

### Copy and fill the `.env` file:
```bash
# From the project root directory
cp .env.template .env
```

### Edit `.env` and fill in your credentials:
```env
# TradeLocker Account Details
TRADELOCKER_SERVER=demo.tradelocker.com  # or live.tradelocker.com
TRADELOCKER_ACCOUNT_ID=YOUR_ACCOUNT_ID_HERE
TRADELOCKER_USERNAME=your.email@example.com
TRADELOCKER_PASSWORD=your_password_here
TRADELOCKER_ACCOUNT_TYPE=demo  # or 'live'

# Account Information
ACCOUNT_SIZE=10000
MAX_DAILY_LOSS=500  # 5% daily loss limit
MAX_TOTAL_DRAWDOWN=800  # 8% total drawdown limit
MAX_POSITION_SIZE=5  # Maximum minis allowed

# Risk Settings
RISK_PER_TRADE_PCT=1.0  # 1% risk per trade
ENABLE_REGIME_ADAPTIVE=true
```

## 3. Verify Configuration

Test that your credentials are loaded correctly:
```bash
python src/config_loader.py
```

You should see:
```
✅ Configuration loaded successfully!
   Server: demo.tradelocker.com
   Account: YOUR_ACCOUNT_ID
   Account Size: $10,000
   Risk per trade: 1.0% ($100)
   Regime adaptive: True
   Instrument: NAS100.mini
```

## 4. Run the Trading Engine (DRY RUN FIRST!)

### Start in monitoring mode:
```bash
python src/live_trading_engine.py
```

### What happens:
- Engine authenticates with TradeLocker
- Displays account balance and equity
- Monitors market during NY session (9:30 AM - 4:00 PM ET)
- Detects opening range breakouts
- **Currently in monitoring mode** - signals logged but NO trades placed yet

### Expected output:
```
================================================================================
🚀 LIVE TRADING ENGINE STARTING
================================================================================
Account: YOUR_ACCOUNT_ID
Account Size: $10,000
Risk per trade: 1.0% ($100)
Max Daily Loss: $500
Max Total Drawdown: $800
Regime Adaptive: True
Instrument: NAS100.mini
✅ TradeLocker authentication successful
Current Balance: $10,000.00
Current Equity: $10,000.00
================================================================================
✅ Engine ready. Entering main loop...
================================================================================
```

## 5. Safety Features

### Automatic Risk Controls:
- ✅ Daily loss limit enforced (default: 5%)
- ✅ Total drawdown limit enforced (default: 8%)
- ✅ Position size capping (default: 5 minis max)
- ✅ Trading hours restriction (NY session only)
- ✅ ATR-based trailing stops (0.75× / 1.0× regime-adaptive)

### Manual Safety:
- Press `Ctrl+C` to gracefully shutdown
- All positions will be closed on shutdown
- All pending orders cancelled

## 6. Next Steps

### Before Going Live:
1. ✅ Run on **DEMO account** for at least 1 week
2. ✅ Verify all signals match backtest expectations
3. ✅ Confirm risk controls work correctly
4. ✅ Test shutdown/restart behavior
5. ✅ Monitor during high volatility days

### When Ready for Live:
1. Change `TRADELOCKER_ACCOUNT_TYPE=live` in `.env`
2. Update credentials to your funded account
3. Start with minimum position sizes
4. Monitor closely for first 2 weeks

## 7. Monitoring and Logs

### Log Files:
- `logs/trading_system.log` - Full detailed logs
- Console output - Real-time status updates

### What to watch:
- Entry signals vs backtest expectations
- Stop loss placement and trailing behavior
- Daily P&L vs risk limits
- ATR regime detection (0.75× vs 1.0×)

## 8. Common Issues

### Authentication Failed:
- Verify credentials in `.env`
- Check server URL (demo vs live)
- Ensure TradeLocker account is active

### No Trades Executing:
- Check if within trading hours (9:30 AM - 4:00 PM ET)
- Verify range is established (after first 5 minutes)
- Check risk limits not breached

### Position Size Too Small:
- Increase `RISK_PER_TRADE_PCT` in `.env`
- Check account size is correct
- Verify ATR calculation

## 9. Support

### Documentation:
- Backtest results: `outputs/orb_full_span_exploration.json`
- Strategy details: See vectorized backtest analysis
- Risk parameters: Optimized from 135 configurations

### Need Help:
- Review logs in `logs/trading_system.log`
- Check AlphaFunded support for TradeLocker issues
- Verify account rules and limits

---

**⚠️ IMPORTANT**: Always start with a DEMO account. Paper trade for at least 1 week before risking real capital.
