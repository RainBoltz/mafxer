# Contributing to MAFXER

Thank you for your interest in contributing to MAFXER! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Submitting Changes](#submitting-changes)
6. [Project Structure](#project-structure)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, background, or identity.

### Expected Behavior

- Be respectful and considerate in all interactions
- Provide constructive feedback
- Focus on the best outcome for the project
- Accept constructive criticism gracefully

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Any conduct that creates an intimidating environment

---

## How Can I Contribute?

### Reporting Bugs

Before submitting a bug report:
- Check if the issue has already been reported
- Verify you're using the latest version
- Test with minimal data to isolate the problem

**Bug Report Template**:
```markdown
**Description**: Brief description of the bug

**Steps to Reproduce**:
1. Step one
2. Step two
3. Step three

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happens

**Environment**:
- Python Version:
- OS:
- Package Versions: (from `pip list`)

**Additional Context**: Any other relevant information
```

### Suggesting Features

We welcome feature suggestions! Please:
- Check if the feature has already been requested
- Clearly describe the use case and benefits
- Consider implementation complexity
- Be open to discussion and alternative approaches

**Feature Request Template**:
```markdown
**Feature Description**: Clear description of the proposed feature

**Use Case**: Why is this feature needed?

**Proposed Solution**: How would this work?

**Alternatives Considered**: Other approaches you've thought about

**Additional Context**: Any other relevant information
```

### Contributing Code

We appreciate code contributions! Areas where you can help:

1. **New Trading Strategies**
   - Implement additional technical indicators (RSI, MACD, Bollinger Bands)
   - Create new trading strategies
   - Add strategy combination logic

2. **Data Processing**
   - Support for different data formats
   - Additional currency pairs
   - Multiple timeframe analysis

3. **Visualization**
   - Interactive charts (plotly, bokeh)
   - Performance dashboards
   - Trade visualization improvements

4. **Performance Metrics**
   - Sharpe ratio calculation
   - Sortino ratio
   - Win rate analysis
   - Drawdown visualization

5. **Testing**
   - Unit tests for trading logic
   - Integration tests
   - Data validation tests

6. **Documentation**
   - Code comments
   - Tutorial improvements
   - Example notebooks
   - Video guides

---

## Development Setup

### Prerequisites

- Python 3.7 or higher
- Git
- pip package manager
- Virtual environment tool (venv or conda)

### Setup Instructions

1. **Fork the repository** on GitHub

2. **Clone your fork**:
```bash
git clone https://github.com/YOUR_USERNAME/mafxer.git
cd mafxer
```

3. **Add upstream remote**:
```bash
git remote add upstream https://github.com/RainBoltz/mafxer.git
```

4. **Create a virtual environment**:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n mafxer python=3.9
conda activate mafxer
```

5. **Install dependencies**:
```bash
pip install -r requirements.txt
```

6. **Verify installation**:
```bash
python simple_ohlc.py
```

### Keeping Your Fork Updated

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

---

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) style guidelines:

- **Indentation**: 4 spaces (no tabs)
- **Line Length**: Maximum 79 characters for code, 72 for comments
- **Naming Conventions**:
  - Functions and variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`

### Code Quality

**Good Example**:
```python
def calculate_sma(prices: pd.Series, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average.

    Args:
        prices: Series of closing prices
        period: Number of periods for SMA calculation

    Returns:
        Array of SMA values with NaN for insufficient data
    """
    result = [np.nan] * len(prices)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1:i + 1])
    return np.array(result)
```

**Avoid**:
```python
# Poor: No documentation, unclear naming
def calc(p, n):
    r = []
    for i in range(len(p)):
        if i < n-1:
            r.append(None)
        else:
            r.append(sum(p[i-n+1:i+1])/n)
    return r
```

### Documentation

**Function Documentation**:
```python
def start_trade(price: np.ndarray, quantity: float, index: int) -> dict:
    """
    Initialize a new trade entry.

    Args:
        price: Array of prices
        quantity: Position size (positive for long, negative for short)
        index: Index in the price array for trade entry

    Returns:
        Dictionary containing trade details:
        - start_price: Entry price
        - start_index: Entry index
        - quantity: Position size
        - end_price: Exit price (None until closed)
        - end_index: Exit index (None until closed)
    """
    return {
        'start_price': price[index],
        'start_index': index,
        'quantity': quantity,
        'end_price': None,
        'end_index': None
    }
```

### Type Hints

Use type hints for better code clarity:

```python
from typing import List, Dict, Tuple, Optional

def analyze_trades(trades: List[Dict]) -> Tuple[float, float, int]:
    """Analyze trading performance."""
    profit: float = 0.0
    loss: float = 0.0
    count: int = 0
    # ... implementation
    return profit, loss, count
```

### Comments

- Use comments to explain **why**, not **what**
- Keep comments up-to-date with code changes
- Avoid obvious comments

**Good**:
```python
# Filter out zero-volume candles as they represent market closures
data = df[df['Volume'] > 0]
```

**Avoid**:
```python
# Set i to 0
i = 0
```

---

## Submitting Changes

### Workflow

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

2. **Make your changes**:
- Write clear, focused commits
- Follow coding standards
- Add tests if applicable
- Update documentation

3. **Test your changes**:
```bash
# Run all scripts to ensure they still work
python simple_ohlc.py
python mono_trade.py
python plot_ohlc.py
python triple_trade.py
```

4. **Commit with clear messages**:
```bash
git add .
git commit -m "Add RSI indicator to trading strategy

- Implement RSI calculation function
- Integrate RSI into signal generation
- Add RSI visualization to charts
- Update documentation with RSI usage"
```

5. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request**:
- Go to GitHub and create a PR from your fork
- Fill in the PR template with details
- Link any related issues
- Request review

### Commit Message Guidelines

**Format**:
```
<type>: <short summary>

<detailed description>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `perf`: Performance improvement
- `style`: Code formatting

**Example**:
```
feat: Add Bollinger Bands indicator

- Implement BB calculation with configurable periods
- Add BB visualization to OHLC charts
- Include BB squeeze detection for volatility analysis

Closes #42
```

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have docstrings
- [ ] Changes are tested and working
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No unnecessary files are included
- [ ] No sensitive data in commits

---

## Project Structure

Understanding the codebase:

```
mafxer/
├── simple_ohlc.py          # Basic OHLC visualization
│   └── Purpose: Display candlestick charts with SMA
│
├── plot_ohlc.py            # Multi-pair visualization
│   └── Purpose: Show correlated pairs with indicators
│
├── mono_trade.py           # Core backtesting engine
│   ├── GO_TRADE(): Main trading logic
│   ├── check_buy(): Buy signal detection
│   ├── check_sell(): Sell signal detection
│   ├── start_trade(): Initialize trade
│   └── close_trade(): Close trade
│
├── triple_trade.py         # Multi-pair strategy
│   └── Purpose: Correlation-based trading
│
├── EURUSD.csv              # Single pair historical data
├── EURTRYUSDTRY.csv        # Multi-pair historical data
├── output.csv              # Backtest results
│
└── Documentation
    ├── README.md           # Project overview
    ├── USAGE.md            # Detailed usage guide
    ├── CONTRIBUTING.md     # This file
    ├── requirements.txt    # Python dependencies
    └── python_class.pdf    # Trading education material
```

### Key Functions to Understand

**Signal Detection**:
```python
def check_buy(fast, slow, i):
    """Detect golden cross (buy signal)"""

def check_sell(fast, slow, i):
    """Detect death cross (sell signal)"""
```

**Trade Management**:
```python
def start_trade(price, quantity, i):
    """Open new position"""

def close_trade(the_trade, price, i):
    """Close existing position"""
```

**Strategy Execution**:
```python
def GO_TRADE(fast_tp, slow_tp):
    """Run backtest with given MA periods"""
```

---

## Testing Guidelines

### Manual Testing

Before submitting changes:

1. **Run all scripts**:
```bash
python simple_ohlc.py    # Should display chart
python plot_ohlc.py      # Should display multi-pair charts
python mono_trade.py     # Should generate output.csv
python triple_trade.py   # Should complete without errors
```

2. **Verify outputs**:
- Charts display correctly
- CSV files are generated
- No errors or warnings
- Results are reasonable

### Adding New Features

When adding features:

1. **Test with edge cases**:
   - Empty data
   - Single data point
   - All buy signals
   - All sell signals
   - No signals

2. **Verify performance**:
   - Large datasets
   - Many parameters
   - Memory usage

3. **Check compatibility**:
   - Different Python versions
   - Different data formats

---

## Getting Help

If you need help:

1. **Read the documentation**:
   - README.md
   - USAGE.md
   - Code comments

2. **Search existing issues**:
   - Check if your question has been answered

3. **Ask questions**:
   - Open a GitHub issue with the "question" label
   - Provide context and what you've tried

4. **Join discussions**:
   - Participate in existing discussions
   - Share your ideas and insights

---

## Recognition

Contributors will be recognized in:
- GitHub contributors page
- Release notes
- Project documentation

Thank you for contributing to MAFXER! Your efforts help make algorithmic trading more accessible to everyone.

---

## License

By contributing to MAFXER, you agree that your contributions will be licensed under the same license as the project.
