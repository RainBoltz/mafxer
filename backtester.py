"""
Backtesting Engine

This module provides a comprehensive backtesting engine for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from strategies import BaseStrategy
from risk_manager import RiskManager, TradeManager
from metrics import PerformanceMetrics


class Backtester:
    """
    Comprehensive backtesting engine for trading strategies.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 strategy: BaseStrategy,
                 initial_capital: float = 100000,
                 quantity: float = 1e5,
                 use_risk_management: bool = False,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04,
                 use_atr_stops: bool = False,
                 use_trailing_stop: bool = False,
                 commission: float = 0.0,
                 slippage: float = 0.0):
        """
        Initialize backtester.

        Args:
            data: DataFrame with OHLC data
            strategy: Trading strategy instance
            initial_capital: Starting capital
            quantity: Position size per trade
            use_risk_management: Whether to use stop-loss/take-profit
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            use_atr_stops: Whether to use ATR-based stops
            use_trailing_stop: Whether to use trailing stops
            commission: Commission per trade (in currency units)
            slippage: Slippage per trade (as price fraction)
        """
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.quantity = quantity
        self.use_risk_management = use_risk_management
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_atr_stops = use_atr_stops
        self.use_trailing_stop = use_trailing_stop
        self.commission = commission
        self.slippage = slippage

        # Initialize risk management
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_position_size=quantity,
            use_atr_stops=use_atr_stops
        )
        self.trade_manager = TradeManager(self.risk_manager)

        # Results
        self.results = None
        self.trades = []
        self.equity_curve = []

    def run(self) -> Dict:
        """
        Run backtest.

        Returns:
            Dictionary with backtest results
        """
        # Generate signals
        print(f"Generating signals for {self.strategy.name}...")
        self.data = self.strategy.generate_signals(self.data)

        # Execute trades
        print(f"Executing trades...")
        self._execute_trades()

        # Calculate performance metrics
        print(f"Calculating performance metrics...")
        metrics_calculator = PerformanceMetrics(self.trades, self.initial_capital)
        metrics = metrics_calculator.calculate_all_metrics()

        # Store results
        self.results = {
            'strategy_name': self.strategy.name,
            'metrics': metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'data': self.data
        }

        return self.results

    def _execute_trades(self) -> None:
        """Execute trades based on signals."""
        current_capital = self.initial_capital
        self.equity_curve = [current_capital]

        for i in range(len(self.data)):
            current_price = self.data['Close'].iloc[i]
            signal = self.data['signal'].iloc[i]

            # Check and close existing positions if stop/take profit hit
            if self.use_risk_management and self.trade_manager.has_open_position():
                closed_trades = self.trade_manager.check_and_close_if_needed(
                    self.data, i, self.use_trailing_stop
                )

                # Update capital and equity curve for closed trades
                for trade in closed_trades:
                    pnl = trade['pnl'] - self.commission
                    # Apply slippage
                    if self.slippage > 0:
                        pnl -= abs(trade['quantity']) * trade['entry_price'] * self.slippage

                    trade['pnl'] = pnl
                    current_capital += pnl
                    self.equity_curve.append(current_capital)

            # Process trading signals
            if signal != 0 and not self.trade_manager.has_open_position():
                # Open new position
                position_quantity = self.quantity if signal > 0 else -self.quantity
                entry_price = self._apply_slippage(current_price, signal)

                # Calculate stop-loss and take-profit
                stop_loss = None
                take_profit = None

                if self.use_risk_management:
                    if self.use_atr_stops:
                        stop_loss, take_profit = self.risk_manager.calculate_atr_stops(
                            self.data, i,
                            'long' if signal > 0 else 'short'
                        )
                    else:
                        stop_loss, take_profit = self.risk_manager.calculate_fixed_stops(
                            entry_price,
                            'long' if signal > 0 else 'short',
                            self.stop_loss_pct,
                            self.take_profit_pct
                        )

                # Open trade
                self.trade_manager.open_trade(
                    entry_price, position_quantity, i,
                    stop_loss, take_profit
                )

            elif signal != 0 and self.trade_manager.has_open_position():
                # Close existing position on opposite signal
                open_trade = self.trade_manager.get_open_position()

                # Check if signal is opposite to current position
                should_close = False
                if open_trade['quantity'] > 0 and signal < 0:  # Long -> Sell signal
                    should_close = True
                elif open_trade['quantity'] < 0 and signal > 0:  # Short -> Buy signal
                    should_close = True

                if should_close:
                    exit_price = self._apply_slippage(current_price, -np.sign(open_trade['quantity']))
                    trade = self.trade_manager.close_trade(open_trade, exit_price, i, 'signal')

                    # Calculate P&L with costs
                    pnl = trade['pnl'] - self.commission
                    if self.slippage > 0:
                        pnl -= abs(trade['quantity']) * trade['entry_price'] * self.slippage

                    trade['pnl'] = pnl
                    self.trades.append(trade)
                    current_capital += pnl
                    self.equity_curve.append(current_capital)

                    # Open new position in opposite direction
                    position_quantity = self.quantity if signal > 0 else -self.quantity
                    entry_price = self._apply_slippage(current_price, signal)

                    stop_loss = None
                    take_profit = None

                    if self.use_risk_management:
                        if self.use_atr_stops:
                            stop_loss, take_profit = self.risk_manager.calculate_atr_stops(
                                self.data, i,
                                'long' if signal > 0 else 'short'
                            )
                        else:
                            stop_loss, take_profit = self.risk_manager.calculate_fixed_stops(
                                entry_price,
                                'long' if signal > 0 else 'short',
                                self.stop_loss_pct,
                                self.take_profit_pct
                            )

                    self.trade_manager.open_trade(
                        entry_price, position_quantity, i,
                        stop_loss, take_profit
                    )

        # Close any remaining open positions
        if self.trade_manager.has_open_position():
            open_trade = self.trade_manager.get_open_position()
            exit_price = self.data['Close'].iloc[-1]
            trade = self.trade_manager.close_trade(open_trade, exit_price, len(self.data) - 1, 'end')

            pnl = trade['pnl'] - self.commission
            if self.slippage > 0:
                pnl -= abs(trade['quantity']) * trade['entry_price'] * self.slippage

            trade['pnl'] = pnl
            self.trades.append(trade)
            current_capital += pnl
            self.equity_curve.append(current_capital)

    def _apply_slippage(self, price: float, signal: int) -> float:
        """
        Apply slippage to execution price.

        Args:
            price: Original price
            signal: Trade direction (1 for buy, -1 for sell)

        Returns:
            Price with slippage applied
        """
        if self.slippage == 0:
            return price

        # Slippage works against you
        if signal > 0:  # Buy - pay higher
            return price * (1 + self.slippage)
        else:  # Sell - receive lower
            return price * (1 - self.slippage)

    def get_results_summary(self) -> Dict:
        """
        Get summary of backtest results.

        Returns:
            Dictionary with summary statistics
        """
        if self.results is None:
            raise ValueError("Backtest has not been run yet. Call run() first.")

        return {
            'strategy': self.strategy.name,
            'total_trades': self.results['metrics']['total_trades'],
            'win_rate': self.results['metrics']['win_rate'],
            'profit_factor': self.results['metrics']['profit_factor'],
            'total_return': self.results['metrics']['total_return'],
            'roi': self.results['metrics']['roi'],
            'max_drawdown_pct': self.results['metrics']['max_drawdown_pct'],
            'sharpe_ratio': self.results['metrics']['sharpe_ratio']
        }

    def print_results(self) -> None:
        """Print detailed backtest results."""
        if self.results is None:
            raise ValueError("Backtest has not been run yet. Call run() first.")

        print(f"\n{'=' * 70}")
        print(f" BACKTEST RESULTS: {self.strategy.name}")
        print(f"{'=' * 70}")

        metrics_calculator = PerformanceMetrics(self.trades, self.initial_capital)
        metrics_calculator.print_summary()


class MultiStrategyBacktester:
    """
    Backtest multiple strategies and compare results.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 strategies: List[BaseStrategy],
                 initial_capital: float = 100000,
                 quantity: float = 1e5,
                 use_risk_management: bool = False,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04):
        """
        Initialize multi-strategy backtester.

        Args:
            data: DataFrame with OHLC data
            strategies: List of strategy instances
            initial_capital: Starting capital
            quantity: Position size per trade
            use_risk_management: Whether to use stop-loss/take-profit
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.data = data
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.quantity = quantity
        self.use_risk_management = use_risk_management
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.results = []

    def run_all(self) -> List[Dict]:
        """
        Run backtest for all strategies.

        Returns:
            List of results dictionaries
        """
        print(f"\n{'=' * 70}")
        print(f" RUNNING MULTI-STRATEGY BACKTEST")
        print(f" Total Strategies: {len(self.strategies)}")
        print(f"{'=' * 70}\n")

        for i, strategy in enumerate(self.strategies, 1):
            print(f"\n[{i}/{len(self.strategies)}] Testing: {strategy.name}")
            print("-" * 70)

            backtester = Backtester(
                self.data,
                strategy,
                self.initial_capital,
                self.quantity,
                self.use_risk_management,
                self.stop_loss_pct,
                self.take_profit_pct
            )

            try:
                result = backtester.run()
                self.results.append(result)
                print(f"✓ Completed: {strategy.name}")
                print(f"  Trades: {result['metrics']['total_trades']}, "
                      f"Win Rate: {result['metrics']['win_rate']:.2f}%, "
                      f"PF: {result['metrics']['profit_factor']:.3f}, "
                      f"Return: {result['metrics']['total_return']:.2f}")
            except Exception as e:
                print(f"✗ Error testing {strategy.name}: {str(e)}")

        return self.results

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get comparison DataFrame of all strategies.

        Returns:
            DataFrame comparing all strategies
        """
        if not self.results:
            raise ValueError("No results available. Run run_all() first.")

        comparison_data = []
        for result in self.results:
            metrics = result['metrics']
            comparison_data.append({
                'Strategy': result['strategy_name'],
                'Trades': metrics['total_trades'],
                'Win Rate %': metrics['win_rate'],
                'Profit Factor': metrics['profit_factor'],
                'Total Return': metrics['total_return'],
                'ROI %': metrics['roi'],
                'Max DD %': metrics['max_drawdown_pct'],
                'Sharpe': metrics['sharpe_ratio'],
                'Sortino': metrics['sortino_ratio'],
                'Expectancy': metrics['expectancy']
            })

        df = pd.DataFrame(comparison_data)
        return df.sort_values('Profit Factor', ascending=False)

    def print_comparison(self) -> None:
        """Print comparison table of all strategies."""
        df = self.get_comparison_dataframe()

        print(f"\n{'=' * 70}")
        print(" STRATEGY COMPARISON")
        print(f"{'=' * 70}\n")
        print(df.to_string(index=False))
        print(f"\n{'=' * 70}\n")

    def get_best_strategy(self, metric: str = 'profit_factor') -> Dict:
        """
        Get best performing strategy.

        Args:
            metric: Metric to use for ranking

        Returns:
            Best strategy result
        """
        if not self.results:
            raise ValueError("No results available. Run run_all() first.")

        best_result = max(self.results, key=lambda x: x['metrics'].get(metric, 0))
        return best_result
