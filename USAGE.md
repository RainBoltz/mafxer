# MAFXER Usage Guide

This guide provides detailed instructions on how to use the MAFXER trading framework for backtesting and analysis.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Scripts Overview](#scripts-overview)
3. [Data Format](#data-format)
4. [Trading Strategy Details](#trading-strategy-details)
5. [Customization](#customization)
6. [Understanding Results](#understanding-results)
7. [Advanced Usage](#advanced-usage)

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RainBoltz/mafxer.git
cd mafxer

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Backtest

```bash
python mono_trade.py
```

This will:
- Load historical EUR/USD data from `EURUSD.csv`
- Test 26 different SMA parameter combinations
- Print performance metrics to console
- Save detailed results to `output.csv`

---

## Scripts Overview

### 1. simple_ohlc.py - Basic Visualization

**Purpose**: Visualize a single currency pair with SMA indicators for a specific time period.

**Usage**:
```bash
python simple_ohlc.py
```

**What it does**:
- Reads EUR/USD data from `EURUSD.csv`
- Filters data for December 14, 2018 (8:00 AM to 9:00 PM)
- Plots candlestick chart with:
  - Fast SMA (13 periods) in purple
  - Slow SMA (26 periods) in green
- Displays interactive matplotlib chart

**Customization**:
Edit the date range in the script:
```python
starttime = pd.to_datetime('2018-12-14 8:00')  # Change start time
deadline = pd.to_datetime('2018-12-14 21:00')   # Change end time
```

Change SMA periods:
```python
data['fast'] = SMA(data['Close'], timeperiod=13)  # Fast MA period
data['slow'] = SMA(data['Close'], timeperiod=26)  # Slow MA period
```

---

### 2. plot_ohlc.py - Multi-Pair Visualization

**Purpose**: Visualize three correlated currency pairs with SMA crossover signals.

**Usage**:
```bash
python plot_ohlc.py
```

**What it does**:
- Loads data for EUR/USD, EUR/TRY, and USD/TRY
- Creates subplots for each pair with SMA indicators
- Highlights crossover points (buy/sell signals)
- Shows correlation between related currency pairs

**Use Case**:
Ideal for understanding how correlated pairs move together and identifying arbitrage opportunities.

---

### 3. mono_trade.py - Backtesting Engine

**Purpose**: Systematic backtesting of SMA crossover strategies with parameter optimization.

**Usage**:
```bash
python mono_trade.py
```

**Configuration**:

The script tests multiple parameter combinations:
```python
test_list = [(i, i*2) for i in range(5, 30+1)]
```

This creates pairs like: (5,10), (6,12), (7,14), ..., (30,60)

**Trading Parameters**:
```python
QUANTITY = 1e5  # Trade size: 100,000 units (1 standard lot)
```

**Output**:

Console output example:
```
(05,10)x234: return=12345, pf=1.234
(06,12)x198: return=15678, pf=1.345
...
```

Where:
- `(05,10)`: Fast MA = 5, Slow MA = 10
- `x234`: Number of trades executed
- `return=12345`: Total return in pipettes
- `pf=1.234`: Profit factor (gross profit / gross loss)

**Results File** (`output.csv`):
```csv
fast_tp,slow_tp,n,return,pf
5,10,234,12345,1.234
6,12,198,15678,1.345
...
```

---

### 4. triple_trade.py - Multi-Pair Strategy

**Purpose**: Advanced trading strategy using correlation between three currency pairs.

**Usage**:
```bash
python triple_trade.py
```

**Strategy Logic**:
- Monitors EUR/USD, EUR/TRY, and USD/TRY simultaneously
- Generates signals based on multiple pair correlations
- More sophisticated than single-pair strategy
- Aims to reduce false signals through cross-pair validation

**Use Case**:
For traders looking to implement more robust strategies using triangular arbitrage or correlation-based signals.

---

## Data Format

### CSV Structure

Your data files should follow this format:

```csv
Gmt time,Open,High,Low,Close,Volume
2018-01-02 00:00:00,1.20150,1.20180,1.20120,1.20150,125
2018-01-02 00:15:00,1.20155,1.20190,1.20130,1.20175,142
```

**Columns**:
- `Gmt time`: Timestamp in GMT timezone
- `Open`: Opening price for the period
- `High`: Highest price during the period
- `Low`: Lowest price during the period
- `Close`: Closing price for the period
- `Volume`: Trading volume (used to filter inactive periods)

**Timeframe**: 15-minute candles (default in provided data)

### Using Your Own Data

To use your own Forex data:

1. Export data in OHLC format from your broker or data provider
2. Ensure column names match the expected format
3. Save as CSV with proper datetime format
4. Update the filename in the scripts:

```python
df = pd.read_csv('YOUR_DATA_FILE.csv')  # Replace with your file
```

---

## Trading Strategy Details

### SMA Crossover Strategy

**Buy Signal (Golden Cross)**:
- Occurs when: Fast MA crosses **above** Slow MA
- Action: Enter long position (buy)
- Implementation:
```python
def check_buy(fast, slow, i):
    return fast[i-1] < slow[i-1] and fast[i] > slow[i]
```

**Sell Signal (Death Cross)**:
- Occurs when: Fast MA crosses **below** Slow MA
- Action: Enter short position (sell)
- Implementation:
```python
def check_sell(fast, slow, i):
    return fast[i-1] > slow[i-1] and fast[i] < slow[i]
```

### Trade Lifecycle

1. **Signal Detection**: Script monitors for MA crossovers
2. **Trade Entry**: Opens position at crossover price
3. **Position Holding**: Maintains position until opposite signal
4. **Trade Exit**: Closes position at next crossover
5. **P&L Calculation**: Computes profit/loss in pipettes

### Trade Structure

Each trade contains:
```python
{
    'start_price': 1.20150,      # Entry price
    'start_index': 234,          # Entry candle index
    'quantity': 100000,          # Position size (+/- for long/short)
    'end_price': 1.20275,        # Exit price
    'end_index': 345             # Exit candle index
}
```

---

## Customization

### Modify SMA Periods

In `mono_trade.py`, adjust the test range:

```python
# Test different ranges
test_list = [(i, i*2) for i in range(10, 50+1)]  # Test 10-50 periods

# Test specific combinations
test_list = [(5,10), (10,20), (13,26), (20,50)]

# Test non-2x ratios
test_list = [(fast, slow) for fast in [5,10,15] for slow in [20,30,50]]
```

### Change Position Size

```python
QUANTITY = 1e5   # Standard lot (100,000 units)
QUANTITY = 1e4   # Mini lot (10,000 units)
QUANTITY = 1e3   # Micro lot (1,000 units)
```

### Add Custom Indicators

You can extend the strategy by adding more technical indicators:

```python
# Add RSI
data['rsi'] = calculate_rsi(data['Close'], 14)

# Add Bollinger Bands
data['bb_upper'], data['bb_lower'] = calculate_bollinger_bands(data['Close'])

# Modify signal logic
def check_buy_with_rsi(fast, slow, rsi, i):
    return (fast[i-1] < slow[i-1] and fast[i] > slow[i]) and rsi[i] < 30
```

### Filter by Time of Day

```python
# Only trade during specific hours
data['hour'] = pd.to_datetime(data['Gmt time']).dt.hour
data = data[(data['hour'] >= 8) & (data['hour'] <= 16)]  # 8 AM - 4 PM
```

---

## Understanding Results

### Performance Metrics

**1. Total Return (pipettes)**
- Raw profit/loss in smallest price units
- For EUR/USD: 1 pip = 0.0001, so 10,000 pipettes = 1 pip = $10 per lot
- Example: `return=21854` means 2.1854 pips or ~$218.54 profit per lot

**2. Profit Factor (PF)**
- Formula: `Gross Profit / Gross Loss`
- **PF > 1.0**: Profitable strategy
- **PF = 1.0**: Break-even
- **PF < 1.0**: Losing strategy
- Example: `pf=1.247` means you make $1.247 for every $1 lost

**3. Number of Trades**
- Total trades executed in the backtest period
- More trades = more data points, but also more transaction costs
- Balance between frequency and quality

**4. ROI (Return on Investment)**
- Percentage return on capital deployed
- Formula: `(Profit - Loss) / Total Cost`
- Accounts for capital efficiency

**5. MDD (Maximum Drawdown)**
- Largest peak-to-trough decline during strategy
- Measures risk and worst-case scenario
- Lower is better (less risk)

### Analyzing output.csv

Open `output.csv` to compare all tested parameters:

```python
import pandas as pd

results = pd.read_csv('output.csv')

# Find best by profit factor
best_pf = results.loc[results['pf'].idxmax()]
print(f"Best PF: {best_pf['fast_tp']}/{best_pf['slow_tp']} = {best_pf['pf']:.3f}")

# Find best by returns
best_ret = results.loc[results['return'].idxmax()]
print(f"Best Return: {best_ret['fast_tp']}/{best_ret['slow_tp']} = {best_ret['return']}")

# Filter profitable strategies
profitable = results[results['pf'] > 1.0]
print(f"\n{len(profitable)} profitable combinations out of {len(results)}")
```

### Interpreting Visualization

When you run `simple_ohlc.py` or `plot_ohlc.py`:

- **Red/Green Candles**: Price movement (green = up, red = down)
- **Purple Line**: Fast moving average (more reactive)
- **Green Line**: Slow moving average (smoother trend)
- **Crossover Points**: Potential buy/sell signals
  - Purple crosses above green = Buy signal
  - Purple crosses below green = Sell signal

---

## Advanced Usage

### Walk-Forward Analysis

Instead of testing on entire dataset:

```python
# Split data into train/test periods
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size:]

# Optimize on training data
# Test optimized parameters on test data
```

### Transaction Cost Analysis

Add spread and commission:

```python
SPREAD = 0.0002  # 2 pip spread
COMMISSION = 5   # $5 per trade

# Adjust profit calculation
diff = (trade['end_price']-trade['start_price'])*trade['quantity']
diff -= (SPREAD * abs(trade['quantity']))  # Subtract spread cost
diff -= COMMISSION  # Subtract commission
```

### Multi-Timeframe Analysis

```python
# Resample to different timeframes
data_15min = df  # Original 15-minute data
data_1hour = df.resample('1H', on='Gmt time').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})
```

### Export Trades for Further Analysis

```python
# Save all trades to CSV
trades_df = pd.DataFrame(trades)
trades_df.to_csv('detailed_trades.csv', index=False)

# Analyze trade duration
trades_df['duration'] = trades_df['end_index'] - trades_df['start_index']
print(f"Average trade duration: {trades_df['duration'].mean()} candles")
```

---

## Tips and Best Practices

### 1. Always Filter Low Volume Periods
```python
data = df[df['Volume'] > 0]  # Remove zero-volume candles
```

### 2. Handle Incomplete Trades
```python
# Remove last trade if still open
if trades and trades[-1]['end_price'] is None:
    trades = trades[:-1]
```

### 3. Account for Slippage
Real-world execution prices differ from backtest prices. Add conservative estimates.

### 4. Avoid Overfitting
- Test on out-of-sample data
- Use simple strategies with few parameters
- Validate across different time periods

### 5. Consider Market Conditions
- Trending markets favor MA crossover strategies
- Ranging markets may generate false signals
- Filter by volatility or ADR (Average Daily Range)

---

## Troubleshooting

### Issue: "File not found" error

**Solution**: Ensure CSV files are in the same directory as the scripts, or provide full path:
```python
df = pd.read_csv('/full/path/to/EURUSD.csv')
```

### Issue: "Module not found" error

**Solution**: Install missing dependencies:
```bash
pip install pandas numpy matplotlib mpl-finance
```

### Issue: Empty charts or no data

**Solution**: Check date range in your data matches the script's date filter:
```python
print(df['Gmt time'].min(), df['Gmt time'].max())  # Check date range
```

### Issue: Poor strategy performance

**Solution**:
- Try different timeframes (1H, 4H, 1D)
- Adjust MA periods based on market volatility
- Add filters (RSI, volume, time of day)
- Consider transaction costs

---

## Further Resources

- **python_class.pdf**: Educational material on Forex trading concepts
- **index.html**: View documentation in browser
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/index.html
- **Technical Analysis**: Study SMA, EMA, RSI, MACD, Bollinger Bands

---

## Getting Help

If you encounter issues:

1. Check this usage guide
2. Review the code comments in the scripts
3. Open an issue on GitHub with:
   - Error message
   - Python version
   - Steps to reproduce

---

**Happy Trading! Remember: Past performance does not guarantee future results. Always practice proper risk management.**
