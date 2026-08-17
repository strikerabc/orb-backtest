"""
live_trading_engine.py — Main automated trading engine for ORB strategy.

Integrates:
- Real-time data from TradeLocker
- Entry signal detection (R-CC, CC, II modes)
- ATR-based trailing stops (0.75× / 1.0× regime-adaptive)
- Position management with correlation tracking
- Risk controls
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List

import pandas as pd

from config_loader import TradingConfig, load_config
from tradelocker_client import TradeLockerClient, Position
from position_manager import PositionManager, ActivePosition


class LiveTradingEngine:
    """Real-time ORB trading engine."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.client = TradeLockerClient(
            server=config.server,
            account_id=config.account_id,
            username=config.username,
            password=config.password
        )
        self.logger = self._setup_logging()
        self.position_manager = PositionManager()

        # State tracking
        self.current_atr: Optional[float] = None
        self.daily_pnl: float = 0.0
        self.total_drawdown: float = 0.0
        self.is_trading_enabled: bool = True
        self.shutdown_requested: bool = False

        # Session state
        self.session_high: Optional[float] = None
        self.session_low: Optional[float] = None
        self.range_established: bool = False
        self.range_timestamp: Optional[datetime] = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning("\n⚠️  Shutdown signal received. Closing positions and exiting...")
        self.shutdown_requested = True

    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("LiveTradingEngine")
        logger.setLevel(getattr(logging, self.config.log_level))

        # File handler
        import os
        os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)

        fh = logging.FileHandler(self.config.log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def start(self) -> None:
        """Start the trading engine."""
        self.logger.info("=" * 80)
        self.logger.info("🚀 LIVE TRADING ENGINE STARTING")
        self.logger.info("=" * 80)
        self.logger.info(f"Account: {self.config.account_id}")
        self.logger.info(f"Account Size: ${self.config.account_size:,.0f}")
        self.logger.info(f"Max Notional Exposure: ${self.config.get_max_notional_exposure():,.0f}")
        self.logger.info(f"Risk Management:")
        self.logger.info(f"  - Base risk per trade: {self.config.risk_per_trade_pct}% (${self.config.get_risk_amount():.0f})")
        self.logger.info(f"  - Correlation scaling: {self.config.enable_correlation_scaling}")
        self.logger.info(f"  - With 1 correlated: ${self.config.get_risk_amount(1):.0f}")
        self.logger.info(f"  - With 2 correlated: ${self.config.get_risk_amount(2):.0f}")
        self.logger.info(f"  - Max Daily Loss: ${self.config.max_daily_loss:.0f}")
        self.logger.info(f"  - Max Total Drawdown: ${self.config.max_total_drawdown:.0f}")
        self.logger.info(f"Strategy:")
        self.logger.info(f"  - Regime Adaptive: {self.config.enable_regime_adaptive}")
        self.logger.info(f"  - ATR 0.75× (low vol) / 1.0× (high vol)")
        self.logger.info(f"  - Breakeven: 0.2R, Trail Start: 0.5R")
        self.logger.info(f"Instruments:")
        self.logger.info(f"  - {self.config.instrument_nas100}: {self.config.nas100_multiplier} bundles/lot")
        self.logger.info(f"  - {self.config.instrument_nas100_mini}: {self.config.nas100_mini_multiplier} bundle/lot")

        # Authenticate
        if not self.client.authenticate():
            self.logger.error("❌ Authentication failed. Exiting.")
            return

        # Get account info
        account_info = self.client.get_account_info()
        if account_info:
            balance = account_info.get("balance", 0)
            equity = account_info.get("equity", 0)
            self.logger.info(f"Current Balance: ${balance:,.2f}")
            self.logger.info(f"Current Equity: ${equity:,.2f}")

        self.logger.info("=" * 80)
        self.logger.info("✅ Engine ready. Entering main loop...")
        self.logger.info("=" * 80)

        # Main trading loop
        self.run_main_loop()

    def run_main_loop(self) -> None:
        """Main event loop - runs continuously during trading hours."""
        loop_count = 0

        while True:
            try:
                loop_count += 1

                # Check if we're in trading hours
                if not self.is_trading_hours():
                    if loop_count % 60 == 1:  # Log once per minute
                        self.logger.info("⏸️  Outside trading hours. Waiting...")
                    time.sleep(60)
                    continue

                # Check risk limits
                if not self.check_risk_limits():
                    self.logger.warning("🛑 Risk limits breached. Trading disabled.")
                    time.sleep(60)
                    continue

                # Update market data
                self.update_market_data()

                # Manage existing positions
                self.manage_positions()

                # Scan for new entry signals
                if self.is_trading_enabled:
                    self.scan_for_entries()

                # Sleep for 1 minute between iterations
                time.sleep(60)

            except KeyboardInterrupt:
                self.logger.info("⛔ Shutdown signal received")
                self.shutdown()
                break
            except Exception as e:
                self.logger.error(f"❌ Error in main loop: {e}", exc_info=True)
                time.sleep(60)

            # Check for shutdown request
            if self.shutdown_requested:
                self.shutdown()
                break

    def is_trading_hours(self) -> bool:
        """Check if current time is within NY session trading hours (UTC)."""
        now = datetime.now(timezone.utc)
        current_time = now.strftime("%H:%M")

        # Parse session hours
        start = self.config.ny_session_start_utc
        end = self.config.ny_session_end_utc

        return start <= current_time <= end

    def check_risk_limits(self) -> bool:
        """Check if risk limits are still within bounds."""
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.config.max_daily_loss:
            self.logger.error(
                f"🛑 Daily loss limit reached: ${self.daily_pnl:.2f} / "
                f"${self.config.max_daily_loss:.2f}"
            )
            self.is_trading_enabled = False
            return False

        # Check total drawdown limit
        if abs(self.total_drawdown) >= self.config.max_total_drawdown:
            self.logger.error(
                f"🛑 Total drawdown limit reached: ${self.total_drawdown:.2f} / "
                f"${self.config.max_total_drawdown:.2f}"
            )
            self.is_trading_enabled = False
            return False

        return True

    def update_market_data(self) -> None:
        """Update market data and calculate ATR."""
        # Get latest price
        symbol = self.config.instrument_nas100_mini
        current_price = self.client.get_latest_price(symbol)

        if current_price is None:
            self.logger.warning("⚠️  Could not get current price")
            return

        # TODO: Calculate real-time ATR from historical data
        # For now, use a placeholder (will implement proper ATR calculation)
        # In production, you'd maintain a rolling window of bars and calculate ATR
        if self.current_atr is None:
            self.current_atr = 50.0  # Placeholder - will be replaced with real calculation

        # Update session range
        self.update_session_range(current_price)

    def update_session_range(self, current_price: float) -> None:
        """Update session high/low for range calculation."""
        now = datetime.now(timezone.utc)

        # Reset range at session start
        session_start_hour = int(self.config.ny_session_start_utc.split(":")[0])
        if now.hour == session_start_hour and now.minute < 5:
            if self.session_high is not None:  # Only reset if we had a previous session
                self.logger.info(
                    f"📊 New session started. Previous range: "
                    f"{self.session_low:.2f} - {self.session_high:.2f}"
                )
            self.session_high = current_price
            self.session_low = current_price
            self.range_established = False
            self.range_timestamp = None

        # Update range
        if self.session_high is None or current_price > self.session_high:
            self.session_high = current_price
        if self.session_low is None or current_price < self.session_low:
            self.session_low = current_price

        # Mark range as established after 5 minutes
        if not self.range_established and self.session_high and self.session_low:
            if now.minute >= 5:
                self.range_established = True
                self.range_timestamp = now
                range_size = self.session_high - self.session_low
                self.logger.info(
                    f"✅ Opening range established: {self.session_low:.2f} - "
                    f"{self.session_high:.2f} (size: {range_size:.2f} pts)"
                )

    def manage_positions(self) -> None:
        """Update trailing stops on existing positions."""
        if not self.position_manager:
            return

        # Get current market prices and update trailing stops
        for position in self.position_manager.get_all_positions():
            current_price = self.client.get_latest_price(position.symbol)

            if current_price is None:
                continue

            # Calculate profit in R
            is_long = position.direction == "long"
            risk = abs(position.entry_price - position.stop_price)

            if is_long:
                profit_r = (current_price - position.entry_price) / risk
            else:
                profit_r = (position.entry_price - current_price) / risk

            # Check for breakeven trigger (0.2 R)
            if not position.breakeven_triggered and profit_r >= position.breakeven_rr:
                position.current_stop = position.entry_price
                position.breakeven_triggered = True

                self.client.modify_position(
                    position.position_id,
                    stop_loss=position.entry_price
                )

                self.logger.info(
                    f"🎯 Breakeven triggered: {position.symbol} @ {profit_r:.2f}R\n"
                    f"   Stop moved to entry: ${position.entry_price:.2f}"
                )

            # Check for trail activation (0.5 R)
            if not position.trailing_active and profit_r >= position.trail_start_rr:
                position.trailing_active = True

                self.logger.info(
                    f"📈 Trailing activated: {position.symbol} @ {profit_r:.2f}R\n"
                    f"   ATR multiplier: {position.atr_multiplier}×"
                )

            # Update trailing stop
            if position.trailing_active:
                trail_distance = position.atr_multiplier * self.current_atr if self.current_atr else 50.0

                if is_long:
                    new_stop = current_price - trail_distance
                    # Only move stop up, never down
                    if new_stop > position.current_stop:
                        position.current_stop = new_stop
                        self.client.modify_position(
                            position.position_id,
                            stop_loss=new_stop
                        )

                        self.logger.info(
                            f"📈 Trail updated: {position.symbol}\n"
                            f"   Current: ${current_price:.2f} ({profit_r:.2f}R)\n"
                            f"   New Stop: ${new_stop:.2f} (trail: {trail_distance:.2f} pts)"
                        )
                else:
                    new_stop = current_price + trail_distance
                    # Only move stop down, never up
                    if new_stop < position.current_stop:
                        position.current_stop = new_stop
                        self.client.modify_position(
                            position.position_id,
                            stop_loss=new_stop
                        )

                        self.logger.info(
                            f"📉 Trail updated: {position.symbol}\n"
                            f"   Current: ${current_price:.2f} ({profit_r:.2f}R)\n"
                            f"   New Stop: ${new_stop:.2f} (trail: {trail_distance:.2f} pts)"
                        )

        # Check for closed positions and update manager
        live_positions = self.client.get_positions()
        live_ids = {pos.id for pos in live_positions}

        for position in list(self.position_manager.get_all_positions()):
            if position.position_id not in live_ids:
                # Position was closed
                removed = self.position_manager.remove_position(position.position_id)
                if removed:
                    self.logger.info(
                        f"🏁 Position closed: {removed.symbol}\n"
                        f"   Remaining positions: {len(self.position_manager)}"
                    )

    def get_atr_multiplier(self) -> float:
        """Get ATR multiplier based on regime (adaptive or static)."""
        if not self.config.enable_regime_adaptive:
            return 0.75

        # Regime threshold: 120 points
        if self.current_atr and self.current_atr >= 120:
            return 1.0  # High volatility
        else:
            return 0.75  # Low volatility

    def scan_for_entries(self) -> None:
        """Scan for new entry signals (R-CC, CC, II modes)."""
        if not self.range_established:
            return

        # Get current price for NAS100 (prefer standard over mini)
        current_price = self.client.get_latest_price(self.config.instrument_nas100)

        if current_price is None:
            return

        # Count correlated positions to adjust risk
        num_correlated = self.position_manager.count_correlated_positions(
            new_signal_mode="CC",  # Will be dynamic based on signal type
            new_symbol=self.config.instrument_nas100
        )

        # Check total exposure limit
        current_exposure = self.position_manager.get_total_exposure()
        max_exposure = self.config.get_max_notional_exposure()

        if current_exposure >= max_exposure * 0.9:  # 90% threshold
            self.logger.warning(
                f"⚠️  Near max exposure: ${current_exposure:,.0f} / ${max_exposure:,.0f}. "
                f"Skipping new entries."
            )
            return

        # TODO: Implement proper signal detection logic from entry_detector.py
        # For now, just check for simple breakouts

        if current_price > self.session_high:
            self.logger.info(
                f"🔔 Bullish breakout detected: {current_price:.2f} > {self.session_high:.2f}"
            )
            self.enter_trade("long", current_price, "CC", num_correlated)

        elif current_price < self.session_low:
            self.logger.info(
                f"🔔 Bearish breakout detected: {current_price:.2f} < {self.session_low:.2f}"
            )
            self.enter_trade("short", current_price, "CC", num_correlated)

    def enter_trade(
        self,
        direction: str,
        entry_price: float,
        mode: str,
        num_correlated: int
    ) -> None:
        """
        Enter a new trade with proper position sizing and risk management.

        Args:
            direction: 'long' or 'short'
            entry_price: Entry price
            mode: Entry mode ('RCC', 'CC', 'II')
            num_correlated: Number of existing correlated positions
        """
        # Calculate position size based on stop distance
        atr_mult = self.get_atr_multiplier()
        stop_distance_points = atr_mult * self.current_atr if self.current_atr else 50.0

        if direction == "long":
            stop_price = entry_price - stop_distance_points
        else:
            stop_price = entry_price + stop_distance_points

        # Get risk amount adjusted for correlation
        risk_amount = self.config.get_risk_amount(num_correlated)

        self.logger.info(
            f"💰 Position Sizing: Risk=${risk_amount:.0f} "
            f"(base ${self.config.get_risk_amount():.0f}, "
            f"{num_correlated} correlated positions)"
        )

        # Choose optimal contract and calculate lot size
        lot_size, symbol, commission = self.config.choose_optimal_contract(
            risk_amount, stop_distance_points, entry_price
        )

        # Calculate notional exposure for this trade
        multiplier = (
            self.config.nas100_multiplier if symbol == self.config.instrument_nas100
            else self.config.nas100_mini_multiplier
        )
        notional = lot_size * multiplier * entry_price

        # Check against remaining exposure capacity
        current_exposure = self.position_manager.get_total_exposure()
        max_exposure = self.config.get_max_notional_exposure()

        if current_exposure + notional > max_exposure:
            self.logger.warning(
                f"⚠️  Trade would exceed max exposure. "
                f"Current: ${current_exposure:,.0f}, "
                f"Trade: ${notional:,.0f}, "
                f"Max: ${max_exposure:,.0f}. Skipping."
            )
            return

        # Calculate target price (1.0 R)
        target_distance_points = abs(entry_price - stop_price)
        if direction == "long":
            target_price = entry_price + target_distance_points
        else:
            target_price = entry_price - target_distance_points

        self.logger.info(
            f"📊 Trade Setup ({mode}):\n"
            f"   Direction: {direction.upper()}\n"
            f"   Symbol: {symbol}\n"
            f"   Lot Size: {lot_size:.3f}\n"
            f"   Entry: ${entry_price:.2f}\n"
            f"   Stop: ${stop_price:.2f} ({stop_distance_points:.2f} pts)\n"
            f"   Target: ${target_price:.2f} (1.0 R)\n"
            f"   Risk: ${risk_amount:.0f}\n"
            f"   Commission: ${commission:.2f}\n"
            f"   Notional: ${notional:,.0f}\n"
            f"   ATR Multiplier: {atr_mult}×"
        )

        # Execute trade via TradeLocker
        side = "buy" if direction == "long" else "sell"
        order_id = self.client.place_order(
            symbol=symbol,
            side=side,
            quantity=lot_size,
            stop_loss=stop_price,
            take_profit=target_price
        )

        if order_id:
            # Track position in manager
            position = ActivePosition(
                position_id=order_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                lot_size=lot_size,
                risk_dollars=risk_amount,
                entry_time=datetime.now(timezone.utc),
                mode=mode,
                current_stop=stop_price,
                atr_multiplier=atr_mult,
                breakeven_rr=0.2,
                trail_start_rr=0.5
            )
            self.position_manager.add_position(position)

            self.logger.info(
                f"✅ Trade executed! Order ID: {order_id}\n"
                f"   Active positions: {len(self.position_manager)}\n"
                f"   Total exposure: ${self.position_manager.get_total_exposure():,.0f}\n"
                f"   Total risk: ${self.position_manager.get_total_risk():.0f}"
            )
        else:
            self.logger.error("❌ Failed to execute trade")

    def shutdown(self) -> None:
        """Graceful shutdown - close all positions and cancel orders."""
        self.logger.info("=" * 80)
        self.logger.info("🛑 SHUTTING DOWN TRADING ENGINE")
        self.logger.info("=" * 80)

        # Close all positions
        positions = self.client.get_positions()
        if positions:
            self.logger.info(f"Closing {len(positions)} open position(s)...")
            for pos in positions:
                self.logger.info(f"  Closing: {pos.symbol}")
                self.client.close_position(pos.id)
        else:
            self.logger.info("No open positions to close")

        # Cancel all orders
        orders = self.client.get_orders()
        if orders:
            self.logger.info(f"Cancelling {len(orders)} pending order(s)...")
            for order in orders:
                self.logger.info(f"  Cancelling: {order.symbol}")
                self.client.cancel_order(order.id)
        else:
            self.logger.info("No pending orders to cancel")

        self.logger.info("=" * 80)
        self.logger.info("✅ Shutdown complete")
        self.logger.info("=" * 80)


def main():
    """Main entry point."""
    try:
        config = load_config()
        engine = LiveTradingEngine(config)
        engine.start()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n📝 Please copy .env.template to .env and fill in your TradeLocker credentials.")
    except KeyboardInterrupt:
        print("\n⛔ Shutdown requested")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
