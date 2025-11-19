"""
Performance Metrics Module

This module provides comprehensive performance analysis metrics
for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class PerformanceMetrics:
    """
    Calculate and analyze trading performance metrics.
    """

    def __init__(self, trades: List[Dict], initial_capital: float = 100000):
        """
        Initialize performance metrics calculator.

        Args:
            trades: List of completed trades
            initial_capital: Starting capital
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self.metrics = {}

    def calculate_all_metrics(self) -> Dict:
        """
        Calculate all performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return self._empty_metrics()

        self.metrics['total_trades'] = len(self.trades)
        self.metrics['winning_trades'] = self._count_winning_trades()
        self.metrics['losing_trades'] = self._count_losing_trades()
        self.metrics['win_rate'] = self._calculate_win_rate()

        self.metrics['total_return'] = self._calculate_total_return()
        self.metrics['gross_profit'] = self._calculate_gross_profit()
        self.metrics['gross_loss'] = self._calculate_gross_loss()
        self.metrics['net_profit'] = self._calculate_net_profit()

        self.metrics['profit_factor'] = self._calculate_profit_factor()
        self.metrics['avg_win'] = self._calculate_average_win()
        self.metrics['avg_loss'] = self._calculate_average_loss()
        self.metrics['largest_win'] = self._calculate_largest_win()
        self.metrics['largest_loss'] = self._calculate_largest_loss()

        self.metrics['max_drawdown'] = self._calculate_max_drawdown()
        self.metrics['max_drawdown_pct'] = self._calculate_max_drawdown_pct()
        self.metrics['recovery_factor'] = self._calculate_recovery_factor()

        self.metrics['sharpe_ratio'] = self._calculate_sharpe_ratio()
        self.metrics['sortino_ratio'] = self._calculate_sortino_ratio()
        self.metrics['calmar_ratio'] = self._calculate_calmar_ratio()

        self.metrics['avg_trade_duration'] = self._calculate_avg_trade_duration()
        self.metrics['roi'] = self._calculate_roi()
        self.metrics['expectancy'] = self._calculate_expectancy()

        self.metrics['consecutive_wins'] = self._calculate_max_consecutive_wins()
        self.metrics['consecutive_losses'] = self._calculate_max_consecutive_losses()

        return self.metrics

    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'net_profit': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'recovery_factor': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'avg_trade_duration': 0.0,
            'roi': 0.0,
            'expectancy': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }

    def _count_winning_trades(self) -> int:
        """Count number of winning trades."""
        return sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)

    def _count_losing_trades(self) -> int:
        """Count number of losing trades."""
        return sum(1 for trade in self.trades if trade.get('pnl', 0) < 0)

    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage."""
        if not self.trades:
            return 0.0
        return (self._count_winning_trades() / len(self.trades)) * 100

    def _calculate_total_return(self) -> float:
        """Calculate total return in currency units."""
        return sum(trade.get('pnl', 0) for trade in self.trades)

    def _calculate_gross_profit(self) -> float:
        """Calculate total profit from winning trades."""
        return sum(trade.get('pnl', 0) for trade in self.trades if trade.get('pnl', 0) > 0)

    def _calculate_gross_loss(self) -> float:
        """Calculate total loss from losing trades."""
        return abs(sum(trade.get('pnl', 0) for trade in self.trades if trade.get('pnl', 0) < 0))

    def _calculate_net_profit(self) -> float:
        """Calculate net profit (gross profit - gross loss)."""
        return self._calculate_gross_profit() - self._calculate_gross_loss()

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_loss = self._calculate_gross_loss()
        if gross_loss == 0:
            return float('inf') if self._calculate_gross_profit() > 0 else 0.0
        return self._calculate_gross_profit() / gross_loss

    def _calculate_average_win(self) -> float:
        """Calculate average winning trade size."""
        winning_trades = [t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0]
        return np.mean(winning_trades) if winning_trades else 0.0

    def _calculate_average_loss(self) -> float:
        """Calculate average losing trade size."""
        losing_trades = [t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0]
        return np.mean(losing_trades) if losing_trades else 0.0

    def _calculate_largest_win(self) -> float:
        """Calculate largest winning trade."""
        winning_trades = [t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0]
        return max(winning_trades) if winning_trades else 0.0

    def _calculate_largest_loss(self) -> float:
        """Calculate largest losing trade."""
        losing_trades = [t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0]
        return min(losing_trades) if losing_trades else 0.0

    def _get_equity_curve(self) -> np.ndarray:
        """Calculate equity curve over time."""
        equity = [self.initial_capital]
        for trade in self.trades:
            equity.append(equity[-1] + trade.get('pnl', 0))
        return np.array(equity)

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown in currency units."""
        equity_curve = self._get_equity_curve()
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = equity_curve - running_max
        return abs(np.min(drawdown))

    def _calculate_max_drawdown_pct(self) -> float:
        """Calculate maximum drawdown as percentage."""
        equity_curve = self._get_equity_curve()
        running_max = np.maximum.accumulate(equity_curve)
        drawdown_pct = (equity_curve - running_max) / running_max
        return abs(np.min(drawdown_pct)) * 100

    def _calculate_recovery_factor(self) -> float:
        """Calculate recovery factor (net profit / max drawdown)."""
        max_dd = self._calculate_max_drawdown()
        if max_dd == 0:
            return float('inf') if self._calculate_net_profit() > 0 else 0.0
        return self._calculate_net_profit() / max_dd

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Sharpe ratio
        """
        returns = np.array([t.get('pnl', 0) for t in self.trades])
        if len(returns) < 2:
            return 0.0

        # Convert to returns percentage
        avg_capital = self.initial_capital + (self._calculate_net_profit() / 2)
        returns_pct = (returns / avg_capital) * 100

        excess_returns = returns_pct - (risk_free_rate / len(returns))
        std_dev = np.std(excess_returns)

        if std_dev == 0:
            return 0.0

        return np.mean(excess_returns) / std_dev * np.sqrt(len(returns))

    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (similar to Sharpe but only considers downside volatility).

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Sortino ratio
        """
        returns = np.array([t.get('pnl', 0) for t in self.trades])
        if len(returns) < 2:
            return 0.0

        # Convert to returns percentage
        avg_capital = self.initial_capital + (self._calculate_net_profit() / 2)
        returns_pct = (returns / avg_capital) * 100

        excess_returns = returns_pct - (risk_free_rate / len(returns))

        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0

        return np.mean(excess_returns) / downside_std * np.sqrt(len(returns))

    def _calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).

        Returns:
            Calmar ratio
        """
        max_dd_pct = self._calculate_max_drawdown_pct()
        if max_dd_pct == 0:
            return float('inf') if self._calculate_roi() > 0 else 0.0

        # Annualized return (assuming trades represent reasonable time period)
        roi = self._calculate_roi()
        return roi / max_dd_pct

    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in bars."""
        durations = []
        for trade in self.trades:
            if trade.get('exit_index') and trade.get('entry_index'):
                duration = trade['exit_index'] - trade['entry_index']
                durations.append(duration)

        return np.mean(durations) if durations else 0.0

    def _calculate_roi(self) -> float:
        """Calculate Return on Investment percentage."""
        net_profit = self._calculate_net_profit()
        return (net_profit / self.initial_capital) * 100

    def _calculate_expectancy(self) -> float:
        """
        Calculate expectancy (average expected profit per trade).

        Returns:
            Expected profit per trade
        """
        if not self.trades:
            return 0.0

        win_rate = self._calculate_win_rate() / 100
        avg_win = self._calculate_average_win()
        avg_loss = abs(self._calculate_average_loss())

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def _calculate_max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning trades."""
        max_consecutive = 0
        current_consecutive = 0

        for trade in self.trades:
            if trade.get('pnl', 0) > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades."""
        max_consecutive = 0
        current_consecutive = 0

        for trade in self.trades:
            if trade.get('pnl', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def get_equity_curve_dataframe(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame.

        Returns:
            DataFrame with equity curve
        """
        equity = self._get_equity_curve()
        df = pd.DataFrame({
            'trade_number': range(len(equity)),
            'equity': equity
        })

        # Calculate drawdown
        running_max = np.maximum.accumulate(equity)
        df['drawdown'] = equity - running_max
        df['drawdown_pct'] = (df['drawdown'] / running_max) * 100

        return df

    def get_trade_analysis_dataframe(self) -> pd.DataFrame:
        """
        Get detailed trade analysis as DataFrame.

        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()

        trade_data = []
        for i, trade in enumerate(self.trades, 1):
            trade_data.append({
                'trade_number': i,
                'entry_price': trade.get('entry_price', 0),
                'exit_price': trade.get('exit_price', 0),
                'quantity': trade.get('quantity', 0),
                'pnl': trade.get('pnl', 0),
                'duration': trade.get('exit_index', 0) - trade.get('entry_index', 0),
                'exit_reason': trade.get('exit_reason', 'unknown'),
                'type': 'long' if trade.get('quantity', 0) > 0 else 'short'
            })

        df = pd.DataFrame(trade_data)
        return df

    def print_summary(self) -> None:
        """Print formatted performance summary."""
        if not self.metrics:
            self.calculate_all_metrics()

        print("\n" + "=" * 70)
        print(" " * 20 + "PERFORMANCE SUMMARY")
        print("=" * 70)

        print(f"\n{'TRADING STATISTICS':<35}")
        print("-" * 70)
        print(f"{'Total Trades:':<35} {self.metrics['total_trades']}")
        print(f"{'Winning Trades:':<35} {self.metrics['winning_trades']}")
        print(f"{'Losing Trades:':<35} {self.metrics['losing_trades']}")
        print(f"{'Win Rate:':<35} {self.metrics['win_rate']:.2f}%")

        print(f"\n{'PROFITABILITY':<35}")
        print("-" * 70)
        print(f"{'Total Return:':<35} {self.metrics['total_return']:.2f}")
        print(f"{'Gross Profit:':<35} {self.metrics['gross_profit']:.2f}")
        print(f"{'Gross Loss:':<35} {self.metrics['gross_loss']:.2f}")
        print(f"{'Net Profit:':<35} {self.metrics['net_profit']:.2f}")
        print(f"{'Profit Factor:':<35} {self.metrics['profit_factor']:.3f}")
        print(f"{'ROI:':<35} {self.metrics['roi']:.2f}%")

        print(f"\n{'TRADE ANALYSIS':<35}")
        print("-" * 70)
        print(f"{'Average Win:':<35} {self.metrics['avg_win']:.2f}")
        print(f"{'Average Loss:':<35} {self.metrics['avg_loss']:.2f}")
        print(f"{'Largest Win:':<35} {self.metrics['largest_win']:.2f}")
        print(f"{'Largest Loss:':<35} {self.metrics['largest_loss']:.2f}")
        print(f"{'Expectancy:':<35} {self.metrics['expectancy']:.2f}")
        print(f"{'Avg Trade Duration (bars):':<35} {self.metrics['avg_trade_duration']:.1f}")

        print(f"\n{'RISK METRICS':<35}")
        print("-" * 70)
        print(f"{'Max Drawdown:':<35} {self.metrics['max_drawdown']:.2f}")
        print(f"{'Max Drawdown %:':<35} {self.metrics['max_drawdown_pct']:.2f}%")
        print(f"{'Recovery Factor:':<35} {self.metrics['recovery_factor']:.3f}")
        print(f"{'Sharpe Ratio:':<35} {self.metrics['sharpe_ratio']:.3f}")
        print(f"{'Sortino Ratio:':<35} {self.metrics['sortino_ratio']:.3f}")
        print(f"{'Calmar Ratio:':<35} {self.metrics['calmar_ratio']:.3f}")

        print(f"\n{'CONSISTENCY':<35}")
        print("-" * 70)
        print(f"{'Max Consecutive Wins:':<35} {self.metrics['consecutive_wins']}")
        print(f"{'Max Consecutive Losses:':<35} {self.metrics['consecutive_losses']}")

        print("=" * 70 + "\n")
