"""
Visualization Module

This module provides visualization functions for backtesting results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from mpl_finance import candlestick_ohlc

plt.style.use('ggplot')


def plot_equity_curve(equity_curve: List[float],
                     title: str = "Equity Curve",
                     save_path: Optional[str] = None):
    """
    Plot equity curve over time.

    Args:
        equity_curve: List of equity values
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(equity_curve, linewidth=2, color='#2E86AB')
    ax.fill_between(range(len(equity_curve)), equity_curve,
                     alpha=0.3, color='#2E86AB')

    ax.set_xlabel('Trade Number', fontsize=12)
    ax.set_ylabel('Equity ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add horizontal line at starting equity
    if equity_curve:
        ax.axhline(y=equity_curve[0], color='gray',
                   linestyle='--', alpha=0.5, label='Initial Capital')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Equity curve saved to {save_path}")

    plt.show()


def plot_drawdown(equity_curve: List[float],
                 title: str = "Drawdown",
                 save_path: Optional[str] = None):
    """
    Plot drawdown chart.

    Args:
        equity_curve: List of equity values
        title: Plot title
        save_path: Path to save figure (optional)
    """
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(range(len(drawdown)), drawdown, 0,
                     where=(drawdown < 0), color='red',
                     alpha=0.3, label='Drawdown')
    ax.plot(drawdown, color='red', linewidth=2)

    ax.set_xlabel('Trade Number', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Drawdown chart saved to {save_path}")

    plt.show()


def plot_strategy_comparison(results: List[Dict],
                            metric: str = 'profit_factor',
                            save_path: Optional[str] = None):
    """
    Plot strategy comparison bar chart.

    Args:
        results: List of backtest results
        metric: Metric to compare
        save_path: Path to save figure (optional)
    """
    strategy_names = [r['strategy_name'] for r in results]
    values = [r['metrics'][metric] for r in results]

    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(range(len(strategy_names)), values,
                  color='#2E86AB', alpha=0.7, edgecolor='black')

    # Color bars based on value
    for i, (bar, val) in enumerate(zip(bars, values)):
        if metric == 'profit_factor':
            if val > 1.5:
                bar.set_color('#06A77D')  # Green for good
            elif val < 1.0:
                bar.set_color('#D62828')  # Red for bad

    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Strategy Comparison: {metric.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(strategy_names)))
    ax.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add reference line
    if metric == 'profit_factor':
        ax.axhline(y=1.0, color='gray', linestyle='--',
                   alpha=0.5, label='Breakeven')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison chart saved to {save_path}")

    plt.show()


def plot_multiple_equity_curves(results: List[Dict],
                                save_path: Optional[str] = None):
    """
    Plot multiple equity curves on same chart.

    Args:
        results: List of backtest results
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    for i, result in enumerate(results):
        equity = result['equity_curve']
        ax.plot(equity, linewidth=2, alpha=0.7,
               label=result['strategy_name'], color=colors[i])

    ax.set_xlabel('Trade Number', fontsize=12)
    ax.set_ylabel('Equity ($)', fontsize=12)
    ax.set_title('Strategy Equity Curves Comparison',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Multi-equity curve saved to {save_path}")

    plt.show()


def plot_returns_distribution(trades: List[Dict],
                             save_path: Optional[str] = None):
    """
    Plot distribution of trade returns.

    Args:
        trades: List of trade dictionaries
        save_path: Path to save figure (optional)
    """
    returns = [t.get('pnl', 0) for t in trades]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1.hist(returns, bins=30, color='#2E86AB',
            alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--',
               alpha=0.5, label='Breakeven')
    ax1.set_xlabel('Trade P&L', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Trade Returns Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(returns, vert=True)
    ax2.axhline(y=0, color='red', linestyle='--',
               alpha=0.5, label='Breakeven')
    ax2.set_ylabel('Trade P&L', fontsize=12)
    ax2.set_title('Trade Returns Box Plot', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Returns distribution saved to {save_path}")

    plt.show()


def plot_strategy_with_signals(data: pd.DataFrame,
                               strategy_name: str,
                               start_idx: int = 0,
                               end_idx: Optional[int] = None,
                               save_path: Optional[str] = None):
    """
    Plot OHLC chart with strategy signals.

    Args:
        data: DataFrame with OHLC and signal data
        strategy_name: Name of strategy
        start_idx: Start index for plotting
        end_idx: End index for plotting
        save_path: Path to save figure (optional)
    """
    if end_idx is None:
        end_idx = min(start_idx + 500, len(data))

    plot_data = data.iloc[start_idx:end_idx].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Prepare OHLC data for candlestick
    if 'Gmt time' in data.columns:
        plot_data['date_num'] = mdates.date2num(plot_data['Gmt time'])
    else:
        plot_data['date_num'] = range(len(plot_data))

    ohlc_data = plot_data[['date_num', 'Open', 'High', 'Low', 'Close']].values

    # Plot candlesticks
    candlestick_ohlc(ax1, ohlc_data, width=0.6, colorup='green',
                     colordown='red', alpha=0.8)

    # Plot indicators if present
    if 'sma_fast' in plot_data.columns:
        ax1.plot(plot_data['date_num'], plot_data['sma_fast'],
                label='Fast MA', linewidth=2, color='purple')
    if 'sma_slow' in plot_data.columns:
        ax1.plot(plot_data['date_num'], plot_data['sma_slow'],
                label='Slow MA', linewidth=2, color='blue')
    if 'ema_fast' in plot_data.columns:
        ax1.plot(plot_data['date_num'], plot_data['ema_fast'],
                label='Fast EMA', linewidth=2, color='purple')
    if 'ema_slow' in plot_data.columns:
        ax1.plot(plot_data['date_num'], plot_data['ema_slow'],
                label='Slow EMA', linewidth=2, color='blue')

    # Plot buy/sell signals
    buy_signals = plot_data[plot_data['signal'] == 1]
    sell_signals = plot_data[plot_data['signal'] == -1]

    ax1.scatter(buy_signals['date_num'], buy_signals['Low'] * 0.999,
               marker='^', color='green', s=100, label='Buy Signal', zorder=5)
    ax1.scatter(sell_signals['date_num'], sell_signals['High'] * 1.001,
               marker='v', color='red', s=100, label='Sell Signal', zorder=5)

    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title(f'{strategy_name} - Signals',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot signal line
    ax2.plot(plot_data['date_num'], plot_data['signal'],
            linewidth=2, color='blue')
    ax2.fill_between(plot_data['date_num'], 0, plot_data['signal'],
                     where=(plot_data['signal'] > 0), color='green',
                     alpha=0.3, label='Buy')
    ax2.fill_between(plot_data['date_num'], 0, plot_data['signal'],
                     where=(plot_data['signal'] < 0), color='red',
                     alpha=0.3, label='Sell')

    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Signal', fontsize=12)
    ax2.set_title('Trading Signals', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Strategy chart saved to {save_path}")

    plt.show()


def create_performance_dashboard(result: Dict,
                                save_path: Optional[str] = None):
    """
    Create comprehensive performance dashboard.

    Args:
        result: Backtest result dictionary
        save_path: Path to save figure (optional)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    equity = result['equity_curve']
    ax1.plot(equity, linewidth=2, color='#2E86AB')
    ax1.fill_between(range(len(equity)), equity, alpha=0.3, color='#2E86AB')
    ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    equity_arr = np.array(equity)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - running_max) / running_max * 100
    ax2.fill_between(range(len(drawdown)), drawdown, 0,
                     where=(drawdown < 0), color='red', alpha=0.3)
    ax2.plot(drawdown, color='red', linewidth=2)
    ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # 3. Returns Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    returns = [t.get('pnl', 0) for t in result['trades']]
    ax3.hist(returns, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Trade P&L')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)

    # 4. Performance Metrics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    metrics = result['metrics']
    metrics_text = f"""
    Strategy: {result['strategy_name']}

    Total Trades: {metrics['total_trades']}
    Win Rate: {metrics['win_rate']:.2f}%
    Profit Factor: {metrics['profit_factor']:.3f}

    Total Return: {metrics['total_return']:.2f}
    ROI: {metrics['roi']:.2f}%
    Max Drawdown: {metrics['max_drawdown_pct']:.2f}%

    Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
    Sortino Ratio: {metrics['sortino_ratio']:.3f}
    Expectancy: {metrics['expectancy']:.2f}
    """

    ax4.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Performance Dashboard: {result["strategy_name"]}',
                fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved to {save_path}")

    plt.show()
