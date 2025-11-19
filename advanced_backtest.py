"""
Advanced Backtesting Script

This script runs comprehensive backtests on multiple trading strategies
and compares their performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from strategies import (
    SMAStrategy, EMAStrategy, RSIStrategy, MACDStrategy,
    BollingerBandsStrategy, CombinedStrategy, TrendFollowingStrategy,
    MeanReversionStrategy
)
from backtester import Backtester, MultiStrategyBacktester
from metrics import PerformanceMetrics

# Configuration
plt.style.use('ggplot')


def load_data(filename: str = 'EURUSD.csv') -> pd.DataFrame:
    """
    Load and prepare OHLC data.

    Args:
        filename: CSV filename

    Returns:
        Prepared DataFrame
    """
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)

    # Parse datetime if present
    if 'Gmt time' in df.columns:
        df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
        df = df.set_index('Gmt time')

    # Filter zero-volume candles
    df = df[df['Volume'] > 0].copy()

    # Reset index
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


def create_strategies() -> list:
    """
    Create list of strategies to test.

    Returns:
        List of strategy instances
    """
    strategies = []

    # 1. SMA Strategies (various periods)
    strategies.append(SMAStrategy(fast_period=5, slow_period=10))
    strategies.append(SMAStrategy(fast_period=10, slow_period=20))
    strategies.append(SMAStrategy(fast_period=13, slow_period=26))
    strategies.append(SMAStrategy(fast_period=20, slow_period=50))

    # 2. EMA Strategies
    strategies.append(EMAStrategy(fast_period=9, slow_period=21))
    strategies.append(EMAStrategy(fast_period=12, slow_period=26))

    # 3. RSI Strategy
    strategies.append(RSIStrategy(period=14, oversold=30, overbought=70))
    strategies.append(RSIStrategy(period=14, oversold=25, overbought=75))

    # 4. MACD Strategy
    strategies.append(MACDStrategy(fast_period=12, slow_period=26, signal_period=9))

    # 5. Bollinger Bands
    strategies.append(BollingerBandsStrategy(period=20, std_dev=2.0))
    strategies.append(BollingerBandsStrategy(period=20, std_dev=2.5))

    # 6. Combined Multi-Indicator Strategy
    strategies.append(CombinedStrategy())

    # 7. Trend Following
    strategies.append(TrendFollowingStrategy(ema_fast=9, ema_slow=21))

    # 8. Mean Reversion
    strategies.append(MeanReversionStrategy())

    return strategies


def run_simple_comparison():
    """Run simple comparison without risk management."""
    print("\n" + "=" * 70)
    print(" SIMPLE STRATEGY COMPARISON (No Risk Management)")
    print("=" * 70)

    # Load data
    data = load_data('EURUSD.csv')

    # Create strategies
    strategies = create_strategies()

    # Run multi-strategy backtest
    multi_backtester = MultiStrategyBacktester(
        data=data,
        strategies=strategies,
        initial_capital=100000,
        quantity=1e5,
        use_risk_management=False
    )

    results = multi_backtester.run_all()

    # Print comparison
    multi_backtester.print_comparison()

    # Save results to CSV
    comparison_df = multi_backtester.get_comparison_dataframe()
    comparison_df.to_csv('strategy_comparison_simple.csv', index=False)
    print("\n✓ Results saved to 'strategy_comparison_simple.csv'")

    return multi_backtester


def run_risk_managed_comparison():
    """Run comparison with risk management."""
    print("\n" + "=" * 70)
    print(" RISK-MANAGED STRATEGY COMPARISON")
    print("=" * 70)

    # Load data
    data = load_data('EURUSD.csv')

    # Create strategies
    strategies = create_strategies()

    # Run multi-strategy backtest with risk management
    multi_backtester = MultiStrategyBacktester(
        data=data,
        strategies=strategies,
        initial_capital=100000,
        quantity=1e5,
        use_risk_management=True,
        stop_loss_pct=0.02,  # 2% stop loss
        take_profit_pct=0.04  # 4% take profit
    )

    results = multi_backtester.run_all()

    # Print comparison
    multi_backtester.print_comparison()

    # Save results to CSV
    comparison_df = multi_backtester.get_comparison_dataframe()
    comparison_df.to_csv('strategy_comparison_risk_managed.csv', index=False)
    print("\n✓ Results saved to 'strategy_comparison_risk_managed.csv'")

    return multi_backtester


def run_parameter_optimization():
    """Optimize SMA strategy parameters."""
    print("\n" + "=" * 70)
    print(" SMA PARAMETER OPTIMIZATION")
    print("=" * 70)

    # Load data
    data = load_data('EURUSD.csv')

    # Test different parameter combinations
    fast_periods = range(5, 31, 1)
    results_data = []

    for fast in fast_periods:
        slow = fast * 2
        strategy = SMAStrategy(fast_period=fast, slow_period=slow)

        backtester = Backtester(
            data=data,
            strategy=strategy,
            initial_capital=100000,
            quantity=1e5,
            use_risk_management=True,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )

        result = backtester.run()
        metrics = result['metrics']

        results_data.append({
            'fast_period': fast,
            'slow_period': slow,
            'trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'total_return': metrics['total_return'],
            'roi': metrics['roi'],
            'max_dd_pct': metrics['max_drawdown_pct'],
            'sharpe': metrics['sharpe_ratio']
        })

        print(f"({fast:02d},{slow:02d}) - Trades: {metrics['total_trades']}, "
              f"Win Rate: {metrics['win_rate']:.2f}%, "
              f"PF: {metrics['profit_factor']:.3f}, "
              f"Return: {metrics['total_return']:.2f}")

    # Save optimization results
    optimization_df = pd.DataFrame(results_data)
    optimization_df.to_csv('sma_optimization_results.csv', index=False)
    print("\n✓ Optimization results saved to 'sma_optimization_results.csv'")

    # Find best parameters
    best_by_pf = optimization_df.loc[optimization_df['profit_factor'].idxmax()]
    best_by_return = optimization_df.loc[optimization_df['total_return'].idxmax()]

    print("\n" + "=" * 70)
    print(" BEST PARAMETERS")
    print("=" * 70)
    print(f"\nBest by Profit Factor:")
    print(f"  Fast: {int(best_by_pf['fast_period'])}, Slow: {int(best_by_pf['slow_period'])}")
    print(f"  PF: {best_by_pf['profit_factor']:.3f}, Return: {best_by_pf['total_return']:.2f}")

    print(f"\nBest by Total Return:")
    print(f"  Fast: {int(best_by_return['fast_period'])}, Slow: {int(best_by_return['slow_period'])}")
    print(f"  PF: {best_by_return['profit_factor']:.3f}, Return: {best_by_return['total_return']:.2f}")

    return optimization_df


def analyze_best_strategy(multi_backtester):
    """Analyze the best performing strategy in detail."""
    print("\n" + "=" * 70)
    print(" DETAILED ANALYSIS OF BEST STRATEGY")
    print("=" * 70)

    # Get best strategy
    best_result = multi_backtester.get_best_strategy('profit_factor')

    print(f"\nBest Strategy: {best_result['strategy_name']}")

    # Print detailed metrics
    metrics_calc = PerformanceMetrics(best_result['trades'], 100000)
    metrics_calc.print_summary()

    # Get trade analysis
    trade_df = metrics_calc.get_trade_analysis_dataframe()
    if not trade_df.empty:
        print("\nFirst 10 Trades:")
        print(trade_df.head(10).to_string(index=False))

        # Save trade details
        trade_df.to_csv(f'{best_result["strategy_name"]}_trades.csv', index=False)
        print(f"\n✓ Trade details saved to '{best_result['strategy_name']}_trades.csv'")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print(" MAFXER - ADVANCED STRATEGY BACKTESTING")
    print("=" * 70)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    try:
        # 1. Simple comparison without risk management
        print("\n\n[1/4] Running simple strategy comparison...")
        simple_results = run_simple_comparison()

        # 2. Comparison with risk management
        print("\n\n[2/4] Running risk-managed strategy comparison...")
        risk_managed_results = run_risk_managed_comparison()

        # 3. Parameter optimization
        print("\n\n[3/4] Running SMA parameter optimization...")
        optimization_results = run_parameter_optimization()

        # 4. Detailed analysis of best strategy
        print("\n\n[4/4] Analyzing best strategy...")
        analyze_best_strategy(risk_managed_results)

        print("\n" + "=" * 70)
        print(" BACKTEST COMPLETE")
        print("=" * 70)
        print(f" Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n✓ All results have been saved to CSV files")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Error during backtest: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
