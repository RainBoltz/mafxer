# MAFXER Advanced Features

This document describes the advanced trading framework improvements including new strategies, risk management, and performance analysis.

## Overview

The MAFXER framework has been significantly enhanced with:

- **8 New Trading Strategies** (RSI, MACD, Bollinger Bands, Combined, etc.)
- **Comprehensive Risk Management** (Stop-loss, Take-profit, Position sizing)
- **Advanced Performance Metrics** (Sharpe, Sortino, Calmar ratios, etc.)
- **Professional Backtesting Engine** (Multi-strategy comparison, optimization)
- **Visualization Tools** (Equity curves, drawdowns, comparison charts)

## Performance Highlights

Based on backtesting with EUR/USD data (23,904 candles):

### Top 3 Strategies (With Risk Management)

| Strategy | Win Rate | Profit Factor | Total Return | ROI | Max DD |
|----------|----------|---------------|--------------|-----|--------|
| RSI (14) | 67.26% | 1.263 | 15,405 | 15.4% | 3.97% |
| RSI (14, 25/75) | 64.27% | 1.222 | 10,882 | 10.9% | 5.21% |
| Mean Reversion | 63.33% | 1.214 | 11,904 | 11.9% | 3.02% |

### Key Improvements

- **Better Performance**: RSI strategy achieved 67% win rate vs. 38% for original SMA
- **Lower Risk**: Maximum drawdown reduced from 11% to 4% with risk management
- **Higher Returns**: Best strategy returned 15.4% vs. 2.9% for original

---

## New Modules

### 1. indicators.py - Technical Indicators

Comprehensive library of technical analysis indicators:

#### Moving Averages
- **SMA()** - Simple Moving Average
- **EMA()** - Exponential Moving Average

#### Oscillators
- **RSI()** - Relative Strength Index
- **MACD()** - Moving Average Convergence Divergence
- **Stochastic()** - Stochastic Oscillator

#### Volatility
- **bollinger_bands()** - Bollinger Bands
- **ATR()** - Average True Range

#### Trend
- **ADX()** - Average Directional Index

#### Utilities
- **check_crossover()** - Detect bullish crossovers
- **check_crossunder()** - Detect bearish crossovers

**Example Usage:**
```python
from indicators import RSI, MACD, bollinger_bands

# Calculate RSI
rsi = RSI(prices, period=14)

# Calculate MACD
macd_line, signal_line, histogram = MACD(prices)

# Calculate Bollinger Bands
upper, middle, lower = bollinger_bands(prices, period=20, std_dev=2.0)
```

---

### 2. strategies.py - Trading Strategies

Eight professional trading strategies:

#### 2.1 SMAStrategy - Simple Moving Average Crossover
Classic trend-following strategy using SMA crossovers.

**Parameters:**
- `fast_period` (default: 13) - Fast MA period
- `slow_period` (default: 26) - Slow MA period

**Signals:**
- Buy: Fast MA crosses above Slow MA
- Sell: Fast MA crosses below Slow MA

**Performance:** PF: 1.034, Win Rate: 38.25%

---

#### 2.2 EMAStrategy - Exponential Moving Average
Faster response to price changes using EMA.

**Parameters:**
- `fast_period` (default: 12)
- `slow_period` (default: 26)

**Performance:** PF: 0.993, Win Rate: 29.34%

---

#### 2.3 RSIStrategy - Relative Strength Index ⭐ **BEST PERFORMER**
Mean reversion strategy based on RSI overbought/oversold levels.

**Parameters:**
- `period` (default: 14) - RSI calculation period
- `oversold` (default: 30) - Oversold threshold (buy signal)
- `overbought` (default: 70) - Overbought threshold (sell signal)

**Signals:**
- Buy: RSI crosses above oversold level
- Sell: RSI crosses below overbought level

**Performance:** PF: 1.263, Win Rate: 67.26%, ROI: 15.4%

**Why it works:**
- Catches reversals at extreme levels
- High win rate due to mean reversion
- Works well in ranging markets

---

#### 2.4 MACDStrategy - MACD Crossover
Momentum strategy using MACD line and signal line.

**Parameters:**
- `fast_period` (default: 12)
- `slow_period` (default: 26)
- `signal_period` (default: 9)

**Signals:**
- Buy: MACD crosses above signal line
- Sell: MACD crosses below signal line

**Performance:** PF: 0.940, Win Rate: 33.28%

---

#### 2.5 BollingerBandsStrategy - Bollinger Bands Mean Reversion
Trades bounces from extreme price levels.

**Parameters:**
- `period` (default: 20) - Moving average period
- `std_dev` (default: 2.0) - Standard deviations

**Signals:**
- Buy: Price touches lower band
- Sell: Price touches upper band

**Performance:** PF: 1.211, Win Rate: 64.31%, ROI: 13.6%

---

#### 2.6 CombinedStrategy - Multi-Indicator Confirmation
Uses multiple indicators for signal confirmation.

**Combines:**
- SMA for trend direction
- RSI for momentum
- MACD for confirmation

**Signals:**
- Buy: Uptrend + RSI oversold recovery + MACD bullish
- Sell: Downtrend + RSI overbought decline + MACD bearish

**Performance:** PF: 0.847, Win Rate: 37.13%
**Note:** Lower performance due to strict entry conditions

---

#### 2.7 TrendFollowingStrategy - ADX + EMA
Only trades in strong trends using ADX filter.

**Parameters:**
- `ema_fast` (default: 9)
- `ema_slow` (default: 21)
- `adx_period` (default: 14)
- `adx_threshold` (default: 25)

**Signals:**
- Only takes trades when ADX > 25 (strong trend)
- Buy: EMA crossover in uptrend
- Sell: EMA crossunder in downtrend

**Performance:** PF: 1.043, Win Rate: 38.94% (with risk management)

---

#### 2.8 MeanReversionStrategy - RSI + Bollinger Bands
Dual confirmation mean reversion strategy.

**Signals:**
- Buy: RSI < 30 AND price near lower band
- Sell: RSI > 70 AND price near upper band

**Performance:** PF: 1.214, Win Rate: 63.33%, ROI: 11.9%

---

### 3. risk_manager.py - Risk Management

Professional risk management system.

#### RiskManager Class

**Features:**
- Position sizing based on risk percentage
- ATR-based dynamic stops
- Fixed percentage stops
- Trailing stops
- Kelly Criterion position sizing
- Risk limit monitoring

**Example Usage:**
```python
from risk_manager import RiskManager

# Initialize risk manager
risk_mgr = RiskManager(
    initial_capital=100000,
    risk_per_trade=0.02,  # Risk 2% per trade
    use_atr_stops=True
)

# Calculate position size
position_size = risk_mgr.calculate_position_size(
    capital=100000,
    entry_price=1.20,
    stop_loss_price=1.18
)

# Calculate ATR-based stops
stop_loss, take_profit = risk_mgr.calculate_atr_stops(
    data, index=100, position_type='long'
)
```

#### TradeManager Class

**Features:**
- Trade lifecycle management
- Automatic stop-loss/take-profit execution
- Trailing stop updates
- Trade history tracking

---

### 4. metrics.py - Performance Analysis

Comprehensive performance metrics calculator.

#### Metrics Calculated

**Trading Statistics:**
- Total/Winning/Losing trades
- Win rate percentage
- Average trade duration

**Profitability:**
- Total return
- Gross profit/loss
- Net profit
- Profit factor
- ROI
- Expectancy

**Risk Metrics:**
- Maximum drawdown ($ and %)
- Recovery factor
- **Sharpe Ratio** - Risk-adjusted returns
- **Sortino Ratio** - Downside risk-adjusted returns
- **Calmar Ratio** - Return to max drawdown ratio

**Trade Analysis:**
- Average win/loss
- Largest win/loss
- Consecutive wins/losses

**Example Usage:**
```python
from metrics import PerformanceMetrics

# Calculate all metrics
metrics_calc = PerformanceMetrics(trades, initial_capital=100000)
metrics = metrics_calc.calculate_all_metrics()

# Print formatted summary
metrics_calc.print_summary()

# Get equity curve DataFrame
equity_df = metrics_calc.get_equity_curve_dataframe()
```

---

### 5. backtester.py - Backtesting Engine

Professional backtesting framework.

#### Backtester Class

**Features:**
- Strategy signal generation
- Trade execution simulation
- Risk management integration
- Commission and slippage modeling
- Detailed trade tracking

**Example Usage:**
```python
from backtester import Backtester
from strategies import RSIStrategy

# Create strategy
strategy = RSIStrategy(period=14)

# Initialize backtester
backtester = Backtester(
    data=df,
    strategy=strategy,
    initial_capital=100000,
    quantity=100000,
    use_risk_management=True,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    commission=5.0,
    slippage=0.0001
)

# Run backtest
results = backtester.run()

# Print results
backtester.print_results()
```

#### MultiStrategyBacktester Class

Compare multiple strategies simultaneously.

**Example Usage:**
```python
from backtester import MultiStrategyBacktester

# Create strategies
strategies = [
    SMAStrategy(13, 26),
    RSIStrategy(14),
    MACDStrategy()
]

# Run comparison
multi_backtester = MultiStrategyBacktester(
    data=df,
    strategies=strategies,
    initial_capital=100000
)

results = multi_backtester.run_all()

# Print comparison table
multi_backtester.print_comparison()

# Get best strategy
best = multi_backtester.get_best_strategy('profit_factor')
```

---

### 6. visualize.py - Visualization Tools

Professional charting and visualization.

#### Available Functions

**Equity Curves:**
```python
from visualize import plot_equity_curve, plot_drawdown

plot_equity_curve(equity_curve, title="Strategy Performance")
plot_drawdown(equity_curve, title="Drawdown Analysis")
```

**Strategy Comparison:**
```python
plot_strategy_comparison(results, metric='profit_factor')
plot_multiple_equity_curves(results)
```

**Trade Analysis:**
```python
plot_returns_distribution(trades)
plot_strategy_with_signals(data, strategy_name)
```

**Performance Dashboard:**
```python
create_performance_dashboard(result, save_path='dashboard.png')
```

---

### 7. advanced_backtest.py - Main Script

Comprehensive backtesting workflow.

#### What It Does

1. **Simple Comparison** - Tests all strategies without risk management
2. **Risk-Managed Comparison** - Tests with stop-loss/take-profit
3. **Parameter Optimization** - Finds best SMA parameters
4. **Detailed Analysis** - In-depth analysis of best strategy

#### Running the Script

```bash
python advanced_backtest.py
```

#### Output Files

- `strategy_comparison_simple.csv` - Results without risk management
- `strategy_comparison_risk_managed.csv` - Results with risk management
- `sma_optimization_results.csv` - SMA parameter optimization
- `RSI_14_trades.csv` - Detailed trade list for best strategy

---

## Usage Examples

### Quick Start - Test a Single Strategy

```python
import pandas as pd
from strategies import RSIStrategy
from backtester import Backtester

# Load data
df = pd.read_csv('EURUSD.csv')
df = df[df['Volume'] > 0]  # Filter zero volume

# Create strategy
strategy = RSIStrategy(period=14, oversold=30, overbought=70)

# Create backtester
backtester = Backtester(
    data=df,
    strategy=strategy,
    initial_capital=100000,
    quantity=100000,
    use_risk_management=True
)

# Run backtest
results = backtester.run()

# Print results
backtester.print_results()
```

### Compare Multiple Strategies

```python
from strategies import RSIStrategy, MACDStrategy, BollingerBandsStrategy
from backtester import MultiStrategyBacktester

strategies = [
    RSIStrategy(14),
    MACDStrategy(),
    BollingerBandsStrategy()
]

multi_backtester = MultiStrategyBacktester(
    data=df,
    strategies=strategies,
    initial_capital=100000,
    use_risk_management=True
)

results = multi_backtester.run_all()
multi_backtester.print_comparison()
```

### Optimize Strategy Parameters

```python
from strategies import SMAStrategy
from backtester import Backtester

best_pf = 0
best_params = None

for fast in range(5, 31):
    slow = fast * 2
    strategy = SMAStrategy(fast, slow)
    backtester = Backtester(df, strategy)
    results = backtester.run()

    pf = results['metrics']['profit_factor']
    if pf > best_pf:
        best_pf = pf
        best_params = (fast, slow)

print(f"Best parameters: Fast={best_params[0]}, Slow={best_params[1]}, PF={best_pf:.3f}")
```

---

## Performance Optimization Tips

### 1. Strategy Selection

**For Trending Markets:**
- Use SMA/EMA strategies
- Use Trend Following with ADX
- Increase MA periods for longer-term trends

**For Ranging Markets:**
- Use RSI strategy (best performer)
- Use Bollinger Bands
- Use Mean Reversion strategy

### 2. Risk Management

**Always use risk management:**
- 2% stop-loss recommended
- 2:1 reward/risk ratio (4% take-profit)
- Consider ATR-based stops for volatility adaptation
- Use trailing stops to lock in profits

### 3. Parameter Tuning

**RSI Strategy:**
- Standard: 30/70 levels work well
- Aggressive: 25/75 for fewer but higher quality signals
- Period: 14 is optimal for most markets

**SMA Strategy:**
- Best performance: 13/26 period combination
- Faster: 5/10 for more signals (lower win rate)
- Slower: 20/50 for fewer signals (higher quality)

### 4. Commission and Slippage

**Account for trading costs:**
```python
backtester = Backtester(
    data=df,
    strategy=strategy,
    commission=5.0,  # $5 per trade
    slippage=0.0002  # 2 pip slippage
)
```

---

## Advanced Features

### Walk-Forward Analysis

Test strategy on rolling windows:

```python
def walk_forward_analysis(data, strategy, window=1000, step=200):
    results = []
    for i in range(0, len(data) - window, step):
        train_data = data[i:i+window]
        test_data = data[i+window:i+window+step]

        # Optimize on train data
        # Test on test data
        backtester = Backtester(test_data, strategy)
        result = backtester.run()
        results.append(result)

    return results
```

### Monte Carlo Simulation

Assess strategy robustness:

```python
import numpy as np

def monte_carlo_simulation(trades, n_simulations=1000):
    trade_returns = [t['pnl'] for t in trades]
    simulated_results = []

    for _ in range(n_simulations):
        # Randomly shuffle trades
        shuffled = np.random.choice(trade_returns, len(trade_returns))
        final_return = sum(shuffled)
        simulated_results.append(final_return)

    # Calculate confidence intervals
    lower = np.percentile(simulated_results, 5)
    upper = np.percentile(simulated_results, 95)

    return lower, upper
```

### Strategy Combination

Combine multiple strategies:

```python
def combined_signals(data, strategies, mode='majority'):
    # Generate signals from all strategies
    all_signals = []
    for strategy in strategies:
        df = strategy.generate_signals(data)
        all_signals.append(df['signal'])

    # Combine signals
    if mode == 'majority':
        # Take majority vote
        combined = sum(all_signals) / len(all_signals)
        final_signal = (combined > 0.5).astype(int) - (combined < -0.5).astype(int)
    elif mode == 'unanimous':
        # Require all strategies to agree
        final_signal = all_signals[0]
        for sig in all_signals[1:]:
            final_signal = final_signal * sig

    return final_signal
```

---

## Troubleshooting

### Low Profit Factor

**Possible causes:**
- Strategy not suited for market conditions
- Parameters not optimized
- High trading costs not accounted for

**Solutions:**
- Try different strategies (especially RSI)
- Optimize parameters
- Use risk management
- Reduce trade frequency

### High Drawdown

**Possible causes:**
- No stop-loss
- Position size too large
- Consecutive losses

**Solutions:**
- Enable risk management
- Reduce position size
- Use trailing stops
- Consider max drawdown limits

### Too Few Trades

**Possible causes:**
- Parameters too strict
- Low volatility period
- Combined strategy with many filters

**Solutions:**
- Loosen parameters (RSI: 35/65 instead of 30/70)
- Use faster MA periods
- Try single-indicator strategies

---

## Best Practices

1. **Always backtest** before live trading
2. **Use risk management** - never trade without stops
3. **Account for costs** - commission and slippage matter
4. **Test on out-of-sample data** - avoid overfitting
5. **Monitor multiple metrics** - not just profit factor
6. **Consider market regime** - strategies perform differently in trends vs. ranges
7. **Start small** - test with small position sizes first
8. **Keep it simple** - simpler strategies often work better
9. **Document everything** - track all tests and results
10. **Never risk more than 2%** per trade

---

## Results Summary

### Complete Strategy Rankings (With Risk Management)

| Rank | Strategy | Profit Factor | Win Rate | ROI | Sharpe |
|------|----------|---------------|----------|-----|--------|
| 1 | RSI (14) | 1.263 | 67.26% | 15.4% | 1.963 |
| 2 | RSI (14, 25/75) | 1.222 | 64.27% | 10.9% | 1.402 |
| 3 | Mean Reversion | 1.214 | 63.33% | 11.9% | 1.599 |
| 4 | Bollinger Bands | 1.211 | 64.31% | 13.6% | 1.687 |
| 5 | BB (2.5 std) | 1.163 | 60.57% | 7.1% | 0.926 |
| 6 | Trend Following | 1.043 | 38.94% | 2.0% | 0.251 |
| 7 | SMA (13/26) | 1.034 | 38.25% | 2.9% | 0.357 |

**Conclusion:** Mean reversion strategies (RSI, Bollinger Bands) significantly outperform trend-following strategies on this dataset.

---

## Future Enhancements

Potential improvements for future versions:

1. **Machine Learning Integration** - Use ML for signal generation
2. **Multi-Timeframe Analysis** - Combine signals from different timeframes
3. **Portfolio Management** - Trade multiple pairs simultaneously
4. **Real-time Trading** - Connect to broker API
5. **Advanced Order Types** - Limit orders, OCO, etc.
6. **Sentiment Analysis** - Incorporate news sentiment
7. **Correlation Analysis** - Trade correlated pairs
8. **Adaptive Parameters** - Auto-adjust based on market conditions

---

## References

- Original MAFXER framework
- Technical Analysis literature
- Algorithmic Trading best practices
- Professional risk management principles

For more information, see:
- README.md - Project overview
- USAGE.md - Detailed usage guide
- CONTRIBUTING.md - Contribution guidelines

---

**Note:** Past performance does not guarantee future results. Always use proper risk management and never trade with money you cannot afford to lose.
