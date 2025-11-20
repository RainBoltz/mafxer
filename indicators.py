"""
Technical Indicators Module

This module provides implementations of various technical analysis indicators
used in trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


def SMA(prices: Union[pd.Series, np.ndarray], period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average.

    Args:
        prices: Series or array of prices
        period: Number of periods for SMA calculation

    Returns:
        Array of SMA values with NaN for insufficient data
    """
    if isinstance(prices, pd.Series):
        return prices.rolling(window=period).mean().values
    else:
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result


def EMA(prices: Union[pd.Series, np.ndarray], period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average.

    Args:
        prices: Series or array of prices
        period: Number of periods for EMA calculation

    Returns:
        Array of EMA values
    """
    if isinstance(prices, pd.Series):
        return prices.ewm(span=period, adjust=False).mean().values
    else:
        ema = np.full(len(prices), np.nan)
        multiplier = 2 / (period + 1)

        # First EMA value is SMA
        ema[period - 1] = np.mean(prices[:period])

        # Calculate subsequent EMA values
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema


def RSI(prices: Union[pd.Series, np.ndarray], period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index.

    Args:
        prices: Series or array of closing prices
        period: Period for RSI calculation (default: 14)

    Returns:
        Array of RSI values (0-100)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)

    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.values


def MACD(prices: Union[pd.Series, np.ndarray],
         fast_period: int = 12,
         slow_period: int = 26,
         signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series or array of closing prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)

    # Calculate MACD line
    ema_fast = EMA(prices, fast_period)
    ema_slow = EMA(prices, slow_period)
    macd_line = ema_fast - ema_slow

    # Calculate signal line
    signal_line = EMA(pd.Series(macd_line), signal_period)

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(prices: Union[pd.Series, np.ndarray],
                   period: int = 20,
                   std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Series or array of closing prices
        period: Period for moving average (default: 20)
        std_dev: Number of standard deviations (default: 2.0)

    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)

    # Calculate middle band (SMA)
    middle_band = prices.rolling(window=period).mean()

    # Calculate standard deviation
    std = prices.rolling(window=period).std()

    # Calculate upper and lower bands
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    return upper_band.values, middle_band.values, lower_band.values


def ATR(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (for volatility measurement).

    Args:
        high: Series or array of high prices
        low: Series or array of low prices
        close: Series or array of closing prices
        period: Period for ATR calculation (default: 14)

    Returns:
        Array of ATR values
    """
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR as EMA of True Range
    atr = tr.ewm(span=period, adjust=False).mean()

    return atr.values


def stochastic_oscillator(high: Union[pd.Series, np.ndarray],
                         low: Union[pd.Series, np.ndarray],
                         close: Union[pd.Series, np.ndarray],
                         k_period: int = 14,
                         d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator.

    Args:
        high: Series or array of high prices
        low: Series or array of low prices
        close: Series or array of closing prices
        k_period: Period for %K calculation (default: 14)
        d_period: Period for %D calculation (default: 3)

    Returns:
        Tuple of (%K, %D)
    """
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)

    # Calculate %K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Calculate %D (SMA of %K)
    d = k.rolling(window=d_period).mean()

    return k.values, d.values


def ADX(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index (trend strength indicator).

    Args:
        high: Series or array of high prices
        low: Series or array of low prices
        close: Series or array of closing prices
        period: Period for ADX calculation (default: 14)

    Returns:
        Array of ADX values
    """
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)

    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate smoothed +DM, -DM, and TR
    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val)

    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    # Calculate ADX as EMA of DX
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx.values


def check_crossover(fast: np.ndarray, slow: np.ndarray, index: int) -> bool:
    """
    Check if fast line crosses above slow line at given index.

    Args:
        fast: Fast moving average array
        slow: Slow moving average array
        index: Index to check crossover

    Returns:
        True if crossover occurred (golden cross)
    """
    if index < 1:
        return False
    return fast[index - 1] < slow[index - 1] and fast[index] > slow[index]


def check_crossunder(fast: np.ndarray, slow: np.ndarray, index: int) -> bool:
    """
    Check if fast line crosses below slow line at given index.

    Args:
        fast: Fast moving average array
        slow: Slow moving average array
        index: Index to check crossunder

    Returns:
        True if crossunder occurred (death cross)
    """
    if index < 1:
        return False
    return fast[index - 1] > slow[index - 1] and fast[index] < slow[index]
