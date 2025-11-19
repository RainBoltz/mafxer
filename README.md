# MAFXER

**M**oving **A**verage **F**ore**X** **E**xchange **R**ate Trading Framework

A Python-based algorithmic trading backtesting and technical analysis framework for Forex currency pairs, focusing on Simple Moving Average (SMA) crossover strategies.

## Overview

MAFXER is an educational and research project that implements automated trading strategy backtesting using historical Forex data. The framework analyzes currency pairs using technical indicators, particularly SMA crossovers, to generate buy/sell signals and evaluate trading performance.

## Features

- **Technical Analysis Visualization**: OHLC (Open-High-Low-Close) candlestick charts with moving average overlays
- **SMA Crossover Strategy**: Automated trading signals based on fast/slow moving average crossings
- **Backtesting Engine**: Test strategies against historical data with configurable parameters
- **Performance Metrics**: Calculate returns, profit factor, ROI, and maximum drawdown
- **Multi-Pair Correlation Trading**: Advanced strategies using correlated currency pairs (EUR/USD, EUR/TRY, USD/TRY)
- **Parameter Optimization**: Test multiple SMA period combinations to find optimal settings

## Project Structure

```
mafxer/
├── simple_ohlc.py         # Basic OHLC visualization for EUR/USD
├── plot_ohlc.py           # Multi-pair visualization with SMA indicators
├── mono_trade.py          # Single-pair backtesting engine
├── triple_trade.py        # Multi-pair correlation trading strategy
├── EURUSD.csv             # Historical EUR/USD 15-minute candle data
├── EURTRYUSDTRY.csv       # Multi-pair historical data (EUR/USD, EUR/TRY, USD/TRY)
├── output.csv             # Backtesting results and performance metrics
├── python_class.pdf       # Educational material on trading concepts
├── index.html             # Web interface for viewing documentation
└── Figure_*.png           # Generated visualization outputs
```

## Technologies

- **Python 3**: Core implementation language
- **Pandas**: Data manipulation and time series analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Chart generation and visualization
- **mpl_finance**: Candlestick chart plotting

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/RainBoltz/mafxer.git
cd mafxer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Visualize Currency Pairs

**Simple single-pair visualization:**
```bash
python simple_ohlc.py
```
Displays EUR/USD candlestick chart for a specific date range with SMA indicators.

**Multi-pair visualization:**
```bash
python plot_ohlc.py
```
Shows correlated currency pairs (EUR/USD, EUR/TRY, USD/TRY) with SMA crossover signals.

### 2. Backtest Trading Strategy

**Single-pair backtesting:**
```bash
python mono_trade.py
```

This script:
- Tests SMA crossover strategies on EUR/USD historical data
- Evaluates multiple parameter combinations (fast_tp: 5-30, slow_tp: fast_tp × 2)
- Calculates performance metrics for each configuration
- Saves results to `output.csv`

**Multi-pair correlation strategy:**
```bash
python triple_trade.py
```

Tests advanced strategies using three correlated currency pairs to generate more robust trading signals.

## Trading Strategy

### SMA Crossover Logic

The framework implements a classic moving average crossover strategy:

1. **Buy Signal**: When fast MA crosses above slow MA (golden cross)
2. **Sell Signal**: When fast MA crosses below slow MA (death cross)

### Performance Metrics

- **Returns**: Total profit/loss in pips
- **Profit Factor (PF)**: Ratio of gross profit to gross loss
- **ROI**: Return on Investment percentage
- **Maximum Drawdown (MDD)**: Largest peak-to-trough decline

## Data Format

The CSV files contain OHLC (Open-High-Low-Close) data with the following structure:

```csv
Date,Open,High,Low,Close,Volume
2018-01-02 00:00:00,1.20150,1.20180,1.20120,1.20150,125
```

- **Timeframe**: 15-minute candles
- **Date Range**: Starting from 2018
- **Pairs**: EUR/USD, EUR/TRY, USD/TRY

## Sample Results

Based on backtesting with historical EUR/USD data, optimal performance was achieved with:

- **Fast MA Period**: 14
- **Slow MA Period**: 28
- **Total Return**: 21,854 pips
- **Profit Factor**: 1.247

See `output.csv` for complete results across 26 different parameter combinations.

## Educational Resources

The `python_class.pdf` file contains comprehensive educational material on:
- Forex trading fundamentals
- Technical analysis concepts
- Moving average strategies
- Risk management principles

View it in your browser by opening `index.html`.

## Contributing

This is an educational and research project. Contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Disclaimer

⚠️ **Important**: This project is for educational and research purposes only. It is not financial advice, and should not be used for live trading without thorough testing and risk assessment. Past performance does not guarantee future results. Trading Forex carries substantial risk of loss.

## License

This project is open source and available for educational purposes.

## Author

RainBoltz

## Acknowledgments

- Historical Forex data providers
- Open source Python trading community
- Technical analysis research community

---

**Note**: Always practice proper risk management and never trade with money you cannot afford to lose.
