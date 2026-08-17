"""
config_loader.py — Secure configuration and credentials loader.

Loads credentials from .env file and provides validated config to the trading system.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class TradingConfig:
    """Trading system configuration."""

    # TradeLocker credentials
    server: str
    account_id: str
    username: str
    password: str
    account_type: str

    # Account settings
    account_size: float
    max_daily_loss: float
    max_total_drawdown: float
    max_position_size: float  # In notional exposure USD

    # Risk management
    risk_per_trade_pct: float
    enable_regime_adaptive: bool
    enable_correlation_scaling: bool  # Scale down risk for correlated positions

    # Contract specifications
    instrument_nas100: str
    instrument_nas100_mini: str
    nas100_multiplier: float  # 100 bundles per lot
    nas100_mini_multiplier: float  # 1 bundle per lot
    nas100_commission: float  # Commission per lot
    nas100_mini_commission: float  # Commission per mini lot
    max_leverage_indices: float  # 25x for indices

    # Trading hours
    ny_session_start_utc: str
    ny_session_end_utc: str

    # Logging
    log_level: str
    log_file: str

    @classmethod
    def load_from_env(cls, env_path: Optional[Path] = None) -> "TradingConfig":
        """Load configuration from .env file."""
        if env_path is None:
            # Try project root first, then src/
            project_root = Path(__file__).resolve().parents[1]
            root_env = project_root / ".env"
            src_env = project_root / "src" / ".env"

            if root_env.exists():
                env_path = root_env
            elif src_env.exists():
                env_path = src_env
            else:
                raise FileNotFoundError(
                    f".env file not found. Tried:\n"
                    f"  - {root_env}\n"
                    f"  - {src_env}\n"
                    f"Copy .env.template to .env and fill in your credentials."
                )

        if not env_path.exists():
            raise FileNotFoundError(
                f".env file not found at {env_path}. "
                f"Copy .env.template to .env and fill in your credentials."
            )

        load_dotenv(env_path)

        # Validate required credentials
        required = [
            "TRADELOCKER_SERVER",
            "TRADELOCKER_ACCOUNT_ID",
            "TRADELOCKER_USERNAME",
            "TRADELOCKER_PASSWORD",
        ]

        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise ValueError(
                f"Missing required credentials in .env: {', '.join(missing)}"
            )

        return cls(
            server=os.getenv("TRADELOCKER_SERVER"),
            account_id=os.getenv("TRADELOCKER_ACCOUNT_ID"),
            username=os.getenv("TRADELOCKER_USERNAME"),
            password=os.getenv("TRADELOCKER_PASSWORD"),
            account_type=os.getenv("TRADELOCKER_ACCOUNT_TYPE", "demo"),
            account_size=float(os.getenv("ACCOUNT_SIZE", "10000")),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "500")),
            max_total_drawdown=float(os.getenv("MAX_TOTAL_DRAWDOWN", "800")),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "250000")),  # Max notional exposure
            risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "1.0")),
            enable_regime_adaptive=os.getenv("ENABLE_REGIME_ADAPTIVE", "true").lower() == "true",
            enable_correlation_scaling=os.getenv("ENABLE_CORRELATION_SCALING", "true").lower() == "true",
            instrument_nas100=os.getenv("INSTRUMENT_NAS100", "NAS100"),
            instrument_nas100_mini=os.getenv("INSTRUMENT_NAS100_MINI", "NAS100.mini"),
            nas100_multiplier=float(os.getenv("NAS100_MULTIPLIER", "100")),  # 100 bundles per lot
            nas100_mini_multiplier=float(os.getenv("NAS100_MINI_MULTIPLIER", "1")),  # 1 bundle per lot
            nas100_commission=float(os.getenv("NAS100_COMMISSION", "8.0")),  # $8 per NAS100 lot
            nas100_mini_commission=float(os.getenv("NAS100_MINI_COMMISSION", "8.0")),  # $8 per mini lot
            max_leverage_indices=float(os.getenv("MAX_LEVERAGE_INDICES", "25")),  # 25x leverage
            ny_session_start_utc=os.getenv("NY_SESSION_START_UTC", "14:30"),
            ny_session_end_utc=os.getenv("NY_SESSION_END_UTC", "21:00"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "logs/trading_system.log"),
        )

    def get_risk_amount(self, num_correlated_positions: int = 0) -> float:
        """
        Calculate dollar risk per trade based on percentage.

        Args:
            num_correlated_positions: Number of existing correlated positions
                                     (reduces risk per trade to account for correlation)

        Returns:
            Dollar risk amount adjusted for correlation if enabled
        """
        base_risk = self.account_size * (self.risk_per_trade_pct / 100.0)

        if not self.enable_correlation_scaling or num_correlated_positions == 0:
            return base_risk

        # Scale down risk for each correlated position
        # With 1 position: 75% of base risk
        # With 2 positions: 60% of base risk
        # With 3+ positions: 50% of base risk (floor)
        scale_factor = max(0.50, 1.0 - (num_correlated_positions * 0.25))

        return base_risk * scale_factor

    def get_max_notional_exposure(self) -> float:
        """Calculate maximum notional exposure based on account size and leverage."""
        return self.account_size * self.max_leverage_indices

    def calculate_lot_size(
        self,
        risk_dollars: float,
        stop_distance_points: float,
        current_price: float,
        use_mini: bool = False
    ) -> tuple[float, str, float]:
        """
        Calculate optimal lot size given risk and stop distance.

        Args:
            risk_dollars: Dollar amount to risk
            stop_distance_points: Stop distance in index points
            current_price: Current market price
            use_mini: Whether to use mini contracts

        Returns:
            (lot_size, instrument_symbol, commission_cost)
        """
        if use_mini:
            multiplier = self.nas100_mini_multiplier
            symbol = self.instrument_nas100_mini
            commission = self.nas100_mini_commission
        else:
            multiplier = self.nas100_multiplier
            symbol = self.instrument_nas100
            commission = self.nas100_commission

        # Dollar value per point = multiplier × 1 point
        dollar_per_point = multiplier

        # Risk per lot = stop_distance_points × dollar_per_point
        risk_per_lot = stop_distance_points * dollar_per_point

        # Lot size = risk_dollars / risk_per_lot
        lot_size = risk_dollars / risk_per_lot

        # Calculate notional exposure
        notional = lot_size * multiplier * current_price

        # Check against max notional
        max_notional = self.get_max_notional_exposure()
        if notional > max_notional:
            lot_size = max_notional / (multiplier * current_price)

        return (lot_size, symbol, commission)

    def choose_optimal_contract(
        self,
        risk_dollars: float,
        stop_distance_points: float,
        current_price: float
    ) -> tuple[float, str, float]:
        """
        Choose between NAS100 and NAS100.mini based on efficiency.

        Returns the contract with lower commission cost relative to position size.
        Prefers NAS100 unless granularity requires mini or cost difference is significant.
        """
        # Calculate both options
        std_lots, std_symbol, std_commission = self.calculate_lot_size(
            risk_dollars, stop_distance_points, current_price, use_mini=False
        )
        mini_lots, mini_symbol, mini_commission = self.calculate_lot_size(
            risk_dollars, stop_distance_points, current_price, use_mini=True
        )

        # For NAS100: 0.05 lot = 5 bundles, commission = $8
        # For NAS100.mini: 5 lots = 5 bundles, commission = 5 × $8 = $40

        # Count number of contracts needed
        std_contracts = max(1, round(std_lots / 0.01))  # NAS100 has 0.01 lot increments
        mini_contracts = max(1, round(mini_lots))  # Mini has 1 lot increments

        std_total_commission = std_contracts * std_commission
        mini_total_commission = mini_contracts * mini_commission

        # Use mini only if it requires fewer contracts AND saves commission
        # OR if standard lot size is too small (< 0.01)
        if std_lots < 0.01:
            return (mini_lots, mini_symbol, mini_total_commission)

        # Default to standard (better commission efficiency for equivalent exposure)
        return (std_lots, std_symbol, std_total_commission)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.account_size <= 0:
            raise ValueError("Account size must be positive")

        if self.max_daily_loss >= self.account_size:
            raise ValueError("Max daily loss must be less than account size")

        if self.max_total_drawdown >= self.account_size:
            raise ValueError("Max total drawdown must be less than account size")

        if self.max_position_size <= 0:
            raise ValueError("Max position size must be positive")

        if not (0 < self.risk_per_trade_pct <= 10):
            raise ValueError("Risk per trade must be between 0 and 10%")

        if self.nas100_multiplier <= 0 or self.nas100_mini_multiplier <= 0:
            raise ValueError("Contract multipliers must be positive")

        if self.max_leverage_indices <= 0:
            raise ValueError("Leverage must be positive")


def load_config(env_path: Optional[Path] = None) -> TradingConfig:
    """Load and validate trading configuration."""
    config = TradingConfig.load_from_env(env_path)
    config.validate()
    return config


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        print("✅ Configuration loaded successfully!")
        print(f"   Server: {config.server}")
        print(f"   Account: {config.account_id}")
        print(f"   Account Size: ${config.account_size:,.0f}")
        print(f"   Max Notional Exposure: ${config.get_max_notional_exposure():,.0f} ({config.max_leverage_indices}x leverage)")
        print(f"\n📊 Risk Management:")
        print(f"   Base risk per trade: {config.risk_per_trade_pct}% (${config.get_risk_amount():.0f})")
        print(f"   With 1 correlated position: ${config.get_risk_amount(1):.0f}")
        print(f"   With 2 correlated positions: ${config.get_risk_amount(2):.0f}")
        print(f"   With 3 correlated positions: ${config.get_risk_amount(3):.0f}")
        print(f"   Correlation scaling: {config.enable_correlation_scaling}")
        print(f"   Regime adaptive: {config.enable_regime_adaptive}")
        print(f"\n📦 Contract Specifications:")
        print(f"   {config.instrument_nas100}: {config.nas100_multiplier} bundles/lot, ${config.nas100_commission} commission")
        print(f"   {config.instrument_nas100_mini}: {config.nas100_mini_multiplier} bundle/lot, ${config.nas100_mini_commission} commission")

        # Example position sizing
        print(f"\n💡 Position Sizing Example (ATR stop = 50 points, Price = 20000):")
        lot_size, symbol, commission = config.choose_optimal_contract(
            config.get_risk_amount(), 50.0, 20000.0
        )
        print(f"   Optimal: {lot_size:.3f} lots of {symbol}, Commission: ${commission:.2f}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
