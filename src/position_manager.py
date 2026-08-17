"""
position_manager.py — Manages active positions and correlation tracking.

Tracks all open positions and calculates correlation exposure for risk scaling.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class ActivePosition:
    """Represents an active trading position."""

    position_id: str
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_price: float
    target_price: float
    lot_size: float
    risk_dollars: float
    entry_time: datetime
    mode: str  # 'RCC', 'CC', 'II'

    # Trailing stop state
    current_stop: float
    trailing_active: bool = False
    breakeven_triggered: bool = False

    # ATR config
    atr_multiplier: float = 0.75
    breakeven_rr: float = 0.2
    trail_start_rr: float = 0.5


class PositionManager:
    """Manages active positions and correlation tracking."""

    def __init__(self):
        self.positions: Dict[str, ActivePosition] = {}

    def add_position(self, position: ActivePosition) -> None:
        """Add a new active position."""
        self.positions[position.position_id] = position

    def remove_position(self, position_id: str) -> ActivePosition | None:
        """Remove and return a closed position."""
        return self.positions.pop(position_id, None)

    def get_position(self, position_id: str) -> ActivePosition | None:
        """Get position by ID."""
        return self.positions.get(position_id)

    def get_all_positions(self) -> List[ActivePosition]:
        """Get all active positions."""
        return list(self.positions.values())

    def count_correlated_positions(self, new_signal_mode: str, new_symbol: str) -> int:
        """
        Count existing positions correlated with a potential new signal.

        All NAS100 positions are considered correlated since they're the same underlying.
        Different modes (RCC, CC, II) on the same instrument are still correlated.

        Args:
            new_signal_mode: Entry mode of the potential new signal
            new_symbol: Symbol of the potential new signal

        Returns:
            Number of existing correlated positions
        """
        if not self.positions:
            return 0

        # All NAS100/NAS100.mini positions are correlated
        nas_symbols = {'NAS100', 'NAS100.mini'}

        correlated_count = 0
        for pos in self.positions.values():
            # Same underlying = correlated
            if pos.symbol in nas_symbols and new_symbol in nas_symbols:
                correlated_count += 1

        return correlated_count

    def get_total_exposure(self) -> float:
        """Calculate total notional exposure across all positions."""
        total = 0.0
        for pos in self.positions.values():
            # NAS100: 100 bundles/lot, NAS100.mini: 1 bundle/lot
            multiplier = 100 if pos.symbol == 'NAS100' else 1
            notional = pos.lot_size * multiplier * pos.entry_price
            total += notional

        return total

    def get_total_risk(self) -> float:
        """Calculate total dollar risk across all positions."""
        return sum(pos.risk_dollars for pos in self.positions.values())

    def get_positions_by_symbol(self, symbol: str) -> List[ActivePosition]:
        """Get all positions for a specific symbol."""
        return [pos for pos in self.positions.values() if pos.symbol == symbol]

    def get_positions_by_mode(self, mode: str) -> List[ActivePosition]:
        """Get all positions for a specific entry mode."""
        return [pos for pos in self.positions.values() if pos.mode == mode]

    def update_stop(self, position_id: str, new_stop: float) -> bool:
        """Update the stop price for a position."""
        pos = self.positions.get(position_id)
        if pos:
            pos.current_stop = new_stop
            return True
        return False

    def mark_breakeven(self, position_id: str) -> bool:
        """Mark a position as having reached breakeven."""
        pos = self.positions.get(position_id)
        if pos:
            pos.breakeven_triggered = True
            pos.current_stop = pos.entry_price
            return True
        return False

    def activate_trailing(self, position_id: str) -> bool:
        """Activate trailing stop for a position."""
        pos = self.positions.get(position_id)
        if pos:
            pos.trailing_active = True
            return True
        return False

    def __len__(self) -> int:
        """Return number of active positions."""
        return len(self.positions)

    def __bool__(self) -> bool:
        """Return True if there are active positions."""
        return len(self.positions) > 0
