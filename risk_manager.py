"""
Risk Management Module

This module provides risk management functionality including
stop-loss, take-profit, position sizing, and risk controls.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from indicators import ATR


class RiskManager:
    """
    Risk management system for trading strategies.

    Handles position sizing, stop-loss, take-profit, and risk controls.
    """

    def __init__(self,
                 initial_capital: float = 100000,
                 risk_per_trade: float = 0.02,
                 max_position_size: float = 1e5,
                 use_atr_stops: bool = False,
                 atr_multiplier: float = 2.0):
        """
        Initialize risk manager.

        Args:
            initial_capital: Starting capital
            risk_per_trade: Risk per trade as fraction of capital (default: 2%)
            max_position_size: Maximum position size
            use_atr_stops: Whether to use ATR-based stops
            atr_multiplier: ATR multiplier for stop-loss (default: 2.0)
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.use_atr_stops = use_atr_stops
        self.atr_multiplier = atr_multiplier

    def calculate_position_size(self,
                               capital: float,
                               entry_price: float,
                               stop_loss_price: Optional[float] = None) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            capital: Current capital
            entry_price: Entry price
            stop_loss_price: Stop loss price (optional)

        Returns:
            Position size in units
        """
        if stop_loss_price is not None:
            # Calculate position size based on risk
            risk_amount = capital * self.risk_per_trade
            price_risk = abs(entry_price - stop_loss_price)

            if price_risk > 0:
                position_size = risk_amount / price_risk
            else:
                position_size = self.max_position_size
        else:
            # Use fixed position size
            position_size = self.max_position_size

        # Cap at maximum position size
        return min(position_size, self.max_position_size)

    def calculate_atr_stops(self,
                           data: pd.DataFrame,
                           index: int,
                           position_type: str,
                           atr_period: int = 14) -> Tuple[float, float]:
        """
        Calculate ATR-based stop-loss and take-profit levels.

        Args:
            data: DataFrame with OHLC data
            index: Current index
            position_type: 'long' or 'short'
            atr_period: ATR period

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Calculate ATR
        if 'atr' not in data.columns:
            data['atr'] = ATR(data['High'], data['Low'], data['Close'], atr_period)

        current_price = data['Close'].iloc[index]
        atr_value = data['atr'].iloc[index]

        if position_type == 'long':
            stop_loss = current_price - (atr_value * self.atr_multiplier)
            take_profit = current_price + (atr_value * self.atr_multiplier * 2)  # 2:1 reward/risk
        else:  # short
            stop_loss = current_price + (atr_value * self.atr_multiplier)
            take_profit = current_price - (atr_value * self.atr_multiplier * 2)

        return stop_loss, take_profit

    def calculate_fixed_stops(self,
                             entry_price: float,
                             position_type: str,
                             stop_loss_pct: float = 0.02,
                             take_profit_pct: float = 0.04) -> Tuple[float, float]:
        """
        Calculate fixed percentage stop-loss and take-profit levels.

        Args:
            entry_price: Entry price
            position_type: 'long' or 'short'
            stop_loss_pct: Stop loss as percentage of entry price
            take_profit_pct: Take profit as percentage of entry price

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if position_type == 'long':
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:  # short
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)

        return stop_loss, take_profit

    def check_stop_loss(self,
                       position: Dict,
                       current_price: float,
                       stop_loss: float) -> bool:
        """
        Check if stop-loss has been hit.

        Args:
            position: Current position dictionary
            current_price: Current market price
            stop_loss: Stop loss price

        Returns:
            True if stop-loss hit
        """
        if position['quantity'] > 0:  # Long position
            return current_price <= stop_loss
        else:  # Short position
            return current_price >= stop_loss

    def check_take_profit(self,
                         position: Dict,
                         current_price: float,
                         take_profit: float) -> bool:
        """
        Check if take-profit has been hit.

        Args:
            position: Current position dictionary
            current_price: Current market price
            take_profit: Take profit price

        Returns:
            True if take-profit hit
        """
        if position['quantity'] > 0:  # Long position
            return current_price >= take_profit
        else:  # Short position
            return current_price <= take_profit

    def apply_trailing_stop(self,
                           position: Dict,
                           current_price: float,
                           highest_price: float,
                           lowest_price: float,
                           trailing_pct: float = 0.02) -> Optional[float]:
        """
        Calculate trailing stop-loss level.

        Args:
            position: Current position dictionary
            current_price: Current market price
            highest_price: Highest price since position opened
            lowest_price: Lowest price since position opened
            trailing_pct: Trailing stop percentage

        Returns:
            Updated stop-loss price or None
        """
        if position['quantity'] > 0:  # Long position
            # Trail stop up as price increases
            new_stop = highest_price * (1 - trailing_pct)
            return new_stop
        else:  # Short position
            # Trail stop down as price decreases
            new_stop = lowest_price * (1 + trailing_pct)
            return new_stop

    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown from equity curve.

        Args:
            equity_curve: Array of equity values over time

        Returns:
            Maximum drawdown as percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)

        # Calculate drawdown at each point
        drawdown = (equity_curve - running_max) / running_max

        # Return maximum drawdown (most negative value)
        max_dd = np.min(drawdown)

        return abs(max_dd)

    def check_risk_limits(self,
                         current_capital: float,
                         max_drawdown: float,
                         max_dd_limit: float = 0.20) -> bool:
        """
        Check if risk limits are breached.

        Args:
            current_capital: Current capital
            max_drawdown: Current maximum drawdown
            max_dd_limit: Maximum allowed drawdown (default: 20%)

        Returns:
            True if within limits, False if limits breached
        """
        # Check if drawdown exceeds limit
        if max_drawdown > max_dd_limit:
            return False

        # Check if capital dropped below critical level
        if current_capital < self.initial_capital * 0.5:  # 50% loss
            return False

        return True

    def calculate_kelly_criterion(self,
                                  win_rate: float,
                                  avg_win: float,
                                  avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            win_rate: Winning percentage (0-1)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size

        Returns:
            Optimal fraction of capital to risk
        """
        if avg_loss == 0 or win_rate == 0:
            return 0.0

        # Kelly formula: f = (p*b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Use fractional Kelly (typically 0.25 or 0.5 of full Kelly) for safety
        fractional_kelly = kelly * 0.25

        # Ensure it's within reasonable bounds
        return max(0.0, min(fractional_kelly, 0.1))  # Cap at 10% of capital


class TradeManager:
    """
    Manages individual trades with risk controls.
    """

    def __init__(self, risk_manager: RiskManager):
        """
        Initialize trade manager.

        Args:
            risk_manager: RiskManager instance
        """
        self.risk_manager = risk_manager
        self.open_trades = []
        self.closed_trades = []

    def open_trade(self,
                   entry_price: float,
                   quantity: float,
                   index: int,
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None) -> Dict:
        """
        Open a new trade.

        Args:
            entry_price: Entry price
            quantity: Position size (positive for long, negative for short)
            index: Entry index
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)

        Returns:
            Trade dictionary
        """
        trade = {
            'entry_price': entry_price,
            'entry_index': index,
            'quantity': quantity,
            'exit_price': None,
            'exit_index': None,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'exit_reason': None
        }

        self.open_trades.append(trade)
        return trade

    def update_trade(self, trade: Dict, current_price: float, index: int) -> None:
        """
        Update trade with current price information.

        Args:
            trade: Trade dictionary
            current_price: Current market price
            index: Current index
        """
        # Update highest/lowest prices for trailing stops
        trade['highest_price'] = max(trade['highest_price'], current_price)
        trade['lowest_price'] = min(trade['lowest_price'], current_price)

    def close_trade(self,
                   trade: Dict,
                   exit_price: float,
                   index: int,
                   reason: str = 'signal') -> Dict:
        """
        Close an open trade.

        Args:
            trade: Trade dictionary
            exit_price: Exit price
            index: Exit index
            reason: Exit reason ('signal', 'stop_loss', 'take_profit', 'trailing_stop')

        Returns:
            Closed trade dictionary
        """
        trade['exit_price'] = exit_price
        trade['exit_index'] = index
        trade['exit_reason'] = reason

        # Calculate P&L
        pnl = (exit_price - trade['entry_price']) * trade['quantity']
        trade['pnl'] = pnl

        # Move from open to closed
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.closed_trades.append(trade)

        return trade

    def check_and_close_if_needed(self,
                                  data: pd.DataFrame,
                                  index: int,
                                  use_trailing_stop: bool = False) -> list:
        """
        Check all open trades for stop-loss or take-profit hits.

        Args:
            data: DataFrame with OHLC data
            index: Current index
            use_trailing_stop: Whether to use trailing stops

        Returns:
            List of closed trades
        """
        closed = []
        current_price = data['Close'].iloc[index]
        high_price = data['High'].iloc[index]
        low_price = data['Low'].iloc[index]

        for trade in self.open_trades[:]:  # Create copy to allow modification
            # Update trade prices
            self.update_trade(trade, current_price, index)

            # Check stop-loss
            if trade['stop_loss'] is not None:
                # Check if stop was hit during the bar
                stop_hit = False
                if trade['quantity'] > 0:  # Long
                    stop_hit = low_price <= trade['stop_loss']
                else:  # Short
                    stop_hit = high_price >= trade['stop_loss']

                if stop_hit:
                    self.close_trade(trade, trade['stop_loss'], index, 'stop_loss')
                    closed.append(trade)
                    continue

            # Check take-profit
            if trade['take_profit'] is not None:
                tp_hit = False
                if trade['quantity'] > 0:  # Long
                    tp_hit = high_price >= trade['take_profit']
                else:  # Short
                    tp_hit = low_price <= trade['take_profit']

                if tp_hit:
                    self.close_trade(trade, trade['take_profit'], index, 'take_profit')
                    closed.append(trade)
                    continue

            # Update trailing stop
            if use_trailing_stop and trade['stop_loss'] is not None:
                new_stop = self.risk_manager.apply_trailing_stop(
                    trade, current_price, trade['highest_price'], trade['lowest_price']
                )
                # Only update if new stop is better
                if trade['quantity'] > 0 and new_stop > trade['stop_loss']:
                    trade['stop_loss'] = new_stop
                elif trade['quantity'] < 0 and new_stop < trade['stop_loss']:
                    trade['stop_loss'] = new_stop

        return closed

    def get_open_position(self) -> Optional[Dict]:
        """
        Get current open position.

        Returns:
            Open trade dictionary or None
        """
        return self.open_trades[0] if self.open_trades else None

    def has_open_position(self) -> bool:
        """
        Check if there's an open position.

        Returns:
            True if position is open
        """
        return len(self.open_trades) > 0
