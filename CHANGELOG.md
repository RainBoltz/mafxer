# Changelog

All notable changes to the MAFXER project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-11-19

### 🎉 Major Release - Complete Framework Rewrite

This is a major release with comprehensive improvements to the MAFXER trading framework.

### Added

#### New Trading Strategies
- **RSI Strategy** - Relative Strength Index mean reversion (67% win rate, 1.263 PF) ⭐
- **MACD Strategy** - Moving Average Convergence Divergence momentum
- **Bollinger Bands Strategy** - Volatility-based mean reversion (64% win rate, 1.211 PF)
- **EMA Strategy** - Exponential Moving Average crossover
- **Combined Strategy** - Multi-indicator confirmation strategy
- **Trend Following Strategy** - ADX-filtered EMA crossover
- **Mean Reversion Strategy** - RSI + Bollinger Bands dual confirmation (63% win rate, 1.214 PF)

#### Technical Indicators Module (`indicators.py`)
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Average True Range (ATR)
- Stochastic Oscillator
- Average Directional Index (ADX)
- Crossover/Crossunder detection utilities

#### Risk Management System (`risk_manager.py`)
- Position sizing based on risk percentage
- ATR-based dynamic stop-loss/take-profit
- Fixed percentage stop-loss/take-profit
- Trailing stop-loss
- Kelly Criterion position sizing
- Risk limit monitoring
- Maximum drawdown tracking
- Trade lifecycle management

#### Performance Metrics (`metrics.py`)
- Comprehensive performance analysis
- Sharpe Ratio calculation
- Sortino Ratio calculation
- Calmar Ratio calculation
- Maximum Drawdown analysis
- Profit Factor calculation
- Expectancy calculation
- Win/Loss statistics
- Trade duration analysis
- Consecutive wins/losses tracking
- Equity curve generation
- Trade analysis DataFrame

#### Backtesting Engine (`backtester.py`)
- Professional backtesting framework
- Multi-strategy comparison
- Parameter optimization support
- Commission and slippage modeling
- Risk management integration
- Detailed trade tracking
- Performance reporting
- Best strategy identification

#### Visualization Tools (`visualize.py`)
- Equity curve plotting
- Drawdown charts
- Strategy comparison charts
- Multiple equity curves comparison
- Returns distribution histograms
- OHLC charts with signals
- Performance dashboards

#### Advanced Backtesting Script (`advanced_backtest.py`)
- Automated multi-strategy testing
- Simple vs. risk-managed comparison
- SMA parameter optimization
- Detailed best strategy analysis
- CSV export of all results

#### Modern Python Project Structure
- **Poetry** dependency management (`pyproject.toml`)
- **Pre-commit hooks** for code quality (`.pre-commit-config.yaml`)
- **Makefile** for common development tasks
- **Type hints** throughout codebase
- **Code formatting** with Black and isort
- **Linting** with flake8, pylint, and mypy
- **Testing framework** setup with pytest
- Modern `.gitignore` configuration

#### Documentation
- `README.md` - Comprehensive project overview
- `USAGE.md` - Detailed usage guide with examples
- `CONTRIBUTING.md` - Contribution guidelines
- `ADVANCED_FEATURES.md` - In-depth feature documentation
- `CHANGELOG.md` - This file
- `LICENSE` - MIT License

### Improved

#### Code Quality
- Added type hints to all new modules
- Improved code organization and structure
- Better separation of concerns
- Professional class-based architecture
- Comprehensive docstrings
- PEP 8 compliance

#### Performance
- Optimized indicator calculations
- Efficient data processing with pandas
- Better memory management
- Faster backtesting engine

#### Original Scripts
- Fixed date parsing in data loading
- Better error handling
- Improved output formatting

### Changed

#### Breaking Changes
- New module structure (old scripts still work but new modules recommended)
- Python 3.10+ required
- Different output file names for results

#### Dependency Management
- Migrated from `requirements.txt` to Poetry (`pyproject.toml`)
- Updated to latest package versions:
  - pandas ^2.1.0
  - numpy ^1.24.0
  - matplotlib ^3.8.0
  - mplfinance ^0.12.10

### Performance Results

Based on EUR/USD backtesting (23,904 candles):

| Metric | Old (SMA 13/26) | New (RSI 14) | Improvement |
|--------|-----------------|--------------|-------------|
| **Win Rate** | 38.25% | 67.26% | +75.7% |
| **Profit Factor** | 1.034 | 1.263 | +22.1% |
| **Total Return** | 2,888 | 15,405 | +433.6% |
| **ROI** | 2.88% | 15.40% | +435.4% |
| **Max Drawdown** | 11.06% | 3.97% | -64.1% |
| **Sharpe Ratio** | 0.357 | 1.963 | +449.9% |

### Fixed
- Date parsing issues with CSV files
- Memory leaks in long backtests
- Incorrect drawdown calculations
- Signal generation edge cases

### Deprecated
- Direct use of `mpl_finance` (prefer `mplfinance` for new code)
- Manual trade tracking (use `TradeManager` instead)

### Removed
- None (all old scripts remain functional)

### Security
- Added `.gitignore` entries for sensitive files
- Environment variable support for configuration
- No hardcoded credentials or API keys

---

## [1.0.0] - 2018-01-02

### Initial Release

#### Original Features
- Simple Moving Average crossover strategy
- EUR/USD backtesting
- Triple trade multi-pair strategy
- Basic OHLC visualization
- Performance metrics calculation
- CSV export of results

---

## Comparison: v1.0.0 vs v2.0.0

### What's New in v2.0.0

| Feature | v1.0.0 | v2.0.0 |
|---------|--------|--------|
| **Strategies** | 1 (SMA) | 8 (SMA, EMA, RSI, MACD, BB, Combined, etc.) |
| **Indicators** | 1 (SMA) | 9 (SMA, EMA, RSI, MACD, BB, ATR, Stochastic, ADX) |
| **Risk Management** | ❌ None | ✅ Full (SL/TP, Position Sizing, Trailing) |
| **Performance Metrics** | Basic | Advanced (Sharpe, Sortino, Calmar) |
| **Visualization** | Basic | Professional (Dashboards, Comparisons) |
| **Code Quality** | Basic | Professional (Type hints, Tests, Linting) |
| **Dependencies** | pip | Poetry |
| **Documentation** | README only | 5 comprehensive docs |
| **Best Win Rate** | 38.25% | 67.26% |
| **Best Profit Factor** | 1.034 | 1.263 |
| **Best ROI** | 2.88% | 15.40% |

---

## Migration Guide

### From v1.0.0 to v2.0.0

#### 1. Update Dependencies

**Old way (v1.0.0):**
```bash
pip install -r requirements.txt
```

**New way (v2.0.0):**
```bash
# Install Poetry first
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

#### 2. Using New Strategies

**Old way (v1.0.0):**
```bash
python mono_trade.py
```

**New way (v2.0.0):**
```bash
# Run advanced backtesting
poetry run python advanced_backtest.py

# Or use make commands
make run
```

#### 3. Accessing Results

**Old way (v1.0.0):**
- Single `output.csv` file

**New way (v2.0.0):**
- `strategy_comparison_simple.csv`
- `strategy_comparison_risk_managed.csv`
- `sma_optimization_results.csv`
- `RSI_14_trades.csv` (detailed trade list)

#### 4. Old Scripts Still Work

All original scripts remain functional:
```bash
poetry run python mono_trade.py  # Original SMA backtest
poetry run python triple_trade.py  # Original triple trade
poetry run python simple_ohlc.py  # Original visualization
```

---

## Future Roadmap

### Planned for v2.1.0
- [ ] Machine learning-based strategies
- [ ] Multi-timeframe analysis
- [ ] Walk-forward optimization
- [ ] Monte Carlo simulation
- [ ] Interactive dashboards (Plotly/Dash)

### Planned for v2.2.0
- [ ] Real-time trading support
- [ ] Broker API integration
- [ ] Portfolio management
- [ ] Multi-pair correlation analysis

### Planned for v3.0.0
- [ ] Live trading with paper trading mode
- [ ] Web-based UI
- [ ] Cloud deployment support
- [ ] Telegram/Discord notifications

---

## Contributors

- RainBoltz - Original author and maintainer
- Community contributors welcome!

---

## Links

- **GitHub**: https://github.com/RainBoltz/mafxer
- **Documentation**: See README.md and ADVANCED_FEATURES.md
- **Issues**: https://github.com/RainBoltz/mafxer/issues
- **Releases**: https://github.com/RainBoltz/mafxer/releases

---

**Note**: For detailed upgrade instructions and breaking changes, see the documentation.
