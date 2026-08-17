"""
tradelocker_client.py — TradeLocker API client for automated trading.

Handles authentication, market data, order execution, and position management.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests


@dataclass
class Position:
    """Open position representation."""
    id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: str
    quantity: float
    order_type: str  # "market", "limit", "stop"
    status: str
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class TradeLockerClient:
    """TradeLocker API client."""

    BASE_URL = "https://api.tradelocker.com"

    def __init__(self, server: str, account_id: str, username: str, password: str):
        self.server = server
        self.account_id = account_id
        self.username = username
        self.password = password
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)

    def authenticate(self) -> bool:
        """Authenticate with TradeLocker and obtain access token."""
        try:
            url = f"{self.BASE_URL}/auth/jwt/token"
            payload = {
                "email": self.username,
                "password": self.password,
                "server": self.server
            }

            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()

            data = response.json()
            self.access_token = data.get("accessToken")
            self.refresh_token = data.get("refreshToken")

            if self.access_token:
                self.session.headers.update({
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                })
                self.logger.info("✅ TradeLocker authentication successful")
                return True
            else:
                self.logger.error("❌ No access token received")
                return False

        except requests.RequestException as e:
            self.logger.error(f"❌ Authentication failed: {e}")
            return False

    def refresh_authentication(self) -> bool:
        """Refresh access token using refresh token."""
        if not self.refresh_token:
            return self.authenticate()

        try:
            url = f"{self.BASE_URL}/auth/jwt/refresh"
            payload = {"refreshToken": self.refresh_token}

            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()

            data = response.json()
            self.access_token = data.get("accessToken")

            if self.access_token:
                self.session.headers.update({
                    "Authorization": f"Bearer {self.access_token}"
                })
                self.logger.info("✅ Token refreshed")
                return True
            else:
                return self.authenticate()

        except requests.RequestException as e:
            self.logger.error(f"❌ Token refresh failed: {e}")
            return self.authenticate()

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information and balance."""
        try:
            url = f"{self.BASE_URL}/backend/accounts/{self.account_id}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to get account info: {e}")
            return None

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        try:
            url = f"{self.BASE_URL}/backend/accounts/{self.account_id}/positions"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            positions_data = response.json()
            positions = []

            for pos in positions_data:
                positions.append(Position(
                    id=pos.get("id"),
                    symbol=pos.get("tradableInstrumentId"),
                    side="buy" if pos.get("qty", 0) > 0 else "sell",
                    quantity=abs(pos.get("qty", 0)),
                    entry_price=pos.get("avgPrice", 0),
                    current_price=pos.get("currentPrice", 0),
                    unrealized_pnl=pos.get("unrealizedPnL", 0),
                    stop_loss=pos.get("stopLoss"),
                    take_profit=pos.get("takeProfit")
                ))

            return positions

        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to get positions: {e}")
            return []

    def get_orders(self) -> List[Order]:
        """Get all pending orders."""
        try:
            url = f"{self.BASE_URL}/backend/accounts/{self.account_id}/orders"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            orders_data = response.json()
            orders = []

            for ord in orders_data:
                orders.append(Order(
                    id=ord.get("id"),
                    symbol=ord.get("tradableInstrumentId"),
                    side=ord.get("side", "").lower(),
                    quantity=ord.get("qty", 0),
                    order_type=ord.get("type", "").lower(),
                    status=ord.get("status", "").lower(),
                    price=ord.get("price"),
                    stop_loss=ord.get("stopLoss"),
                    take_profit=ord.get("takeProfit")
                ))

            return orders

        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to get orders: {e}")
            return []

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[str]:
        """
        Place an order.

        Args:
            symbol: Instrument symbol (e.g., "NAS100.mini")
            side: "buy" or "sell"
            quantity: Number of contracts (can be fractional for minis)
            order_type: "market", "limit", or "stop"
            price: Limit/stop price (required for non-market orders)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            url = f"{self.BASE_URL}/backend/accounts/{self.account_id}/orders"

            payload = {
                "tradableInstrumentId": symbol,
                "side": side.lower(),
                "qty": quantity,
                "type": order_type.lower()
            }

            if price is not None:
                payload["price"] = price
            if stop_loss is not None:
                payload["stopLoss"] = stop_loss
            if take_profit is not None:
                payload["takeProfit"] = take_profit

            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()

            order_data = response.json()
            order_id = order_data.get("orderId")

            self.logger.info(
                f"✅ Order placed: {side.upper()} {quantity} {symbol} @ "
                f"{order_type.upper()} (ID: {order_id})"
            )

            return order_id

        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to place order: {e}")
            return None

    def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Modify an existing position's stop loss or take profit.

        Args:
            position_id: Position ID
            stop_loss: New stop loss price
            take_profit: New take profit price

        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.BASE_URL}/backend/accounts/{self.account_id}/positions/{position_id}"

            payload = {}
            if stop_loss is not None:
                payload["stopLoss"] = stop_loss
            if take_profit is not None:
                payload["takeProfit"] = take_profit

            if not payload:
                self.logger.warning("No modifications specified")
                return False

            response = self.session.patch(url, json=payload, timeout=10)
            response.raise_for_status()

            self.logger.info(f"✅ Position {position_id} modified: SL={stop_loss}, TP={take_profit}")
            return True

        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to modify position: {e}")
            return False

    def close_position(self, position_id: str) -> bool:
        """Close an open position."""
        try:
            url = f"{self.BASE_URL}/backend/accounts/{self.account_id}/positions/{position_id}"
            response = self.session.delete(url, timeout=10)
            response.raise_for_status()

            self.logger.info(f"✅ Position {position_id} closed")
            return True

        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to close position: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            url = f"{self.BASE_URL}/backend/accounts/{self.account_id}/orders/{order_id}"
            response = self.session.delete(url, timeout=10)
            response.raise_for_status()

            self.logger.info(f"✅ Order {order_id} cancelled")
            return True

        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to cancel order: {e}")
            return False

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest market price for symbol."""
        try:
            url = f"{self.BASE_URL}/backend/instruments/quotes"
            params = {"tradableInstrumentIds": symbol}

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            quotes = response.json()
            if quotes and len(quotes) > 0:
                quote = quotes[0]
                # Use mid price between bid and ask
                bid = quote.get("bid", 0)
                ask = quote.get("ask", 0)
                return (bid + ask) / 2.0

            return None

        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to get price: {e}")
            return None
