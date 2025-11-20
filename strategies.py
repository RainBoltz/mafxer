"""
Trading Strategies Module

This module contains various trading strategy implementations.
Each strategy generates buy/sell signals based on technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from indicators import (
    SMA, EMA, RSI, MACD, bollinger_bands, ATR, stochastic_oscillator, ADX,
    check_crossover, check_crossunder
)


class BaseStrategy:
    """Base class for all trading strategies."""

    def __init__(self, name: str):
        """
        Initialize strategy.

        Args:
            name: Strategy name
        """
        self.name = name

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with added signal columns
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")


class SMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.

    Buy when fast MA crosses above slow MA.
    Sell when fast MA crosses below slow MA.
    """

    def __init__(self, fast_period: int = 13, slow_period: int = 26):
        """
        Initialize SMA strategy.

        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
        """
        super().__init__(f"SMA_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on SMA crossover."""
        df = data.copy()

        # Calculate indicators
        df['sma_fast'] = SMA(df['Close'], self.fast_period)
        df['sma_slow'] = SMA(df['Close'], self.slow_period)

        # Generate signals
        df['signal'] = 0
        for i in range(1, len(df)):
            if check_crossover(df['sma_fast'].values, df['sma_slow'].values, i):
                df.loc[df.index[i], 'signal'] = 1  # Buy
            elif check_crossunder(df['sma_fast'].values, df['sma_slow'].values, i):
                df.loc[df.index[i], 'signal'] = -1  # Sell

        return df


class EMAStrategy(BaseStrategy):
    """
    Exponential Moving Average Crossover Strategy.

    Similar to SMA but uses EMA for faster response to price changes.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        """
        Initialize EMA strategy.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
        """
        super().__init__(f"EMA_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on EMA crossover."""
        df = data.copy()

        # Calculate indicators
        df['ema_fast'] = EMA(df['Close'], self.fast_period)
        df['ema_slow'] = EMA(df['Close'], self.slow_period)

        # Generate signals
        df['signal'] = 0
        for i in range(1, len(df)):
            if check_crossover(df['ema_fast'].values, df['ema_slow'].values, i):
                df.loc[df.index[i], 'signal'] = 1  # Buy
            elif check_crossunder(df['ema_fast'].values, df['ema_slow'].values, i):
                df.loc[df.index[i], 'signal'] = -1  # Sell

        return df


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) Strategy.

    Buy when RSI crosses above oversold level (30).
    Sell when RSI crosses below overbought level (70).
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize RSI strategy.

        Args:
            period: RSI period
            oversold: Oversold threshold (buy signal)
            overbought: Overbought threshold (sell signal)
        """
        super().__init__(f"RSI_{period}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on RSI levels."""
        df = data.copy()

        # Calculate RSI
        df['rsi'] = RSI(df['Close'], self.period)

        # Generate signals
        df['signal'] = 0
        for i in range(1, len(df)):
            # Buy when RSI crosses above oversold
            if df['rsi'].iloc[i - 1] < self.oversold and df['rsi'].iloc[i] >= self.oversold:
                df.loc[df.index[i], 'signal'] = 1
            # Sell when RSI crosses below overbought
            elif df['rsi'].iloc[i - 1] > self.overbought and df['rsi'].iloc[i] <= self.overbought:
                df.loc[df.index[i], 'signal'] = -1

        return df


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.

    Buy when MACD line crosses above signal line.
    Sell when MACD line crosses below signal line.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD strategy.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        """
        super().__init__(f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on MACD crossover."""
        df = data.copy()

        # Calculate MACD
        macd_line, signal_line, histogram = MACD(
            df['Close'], self.fast_period, self.slow_period, self.signal_period
        )
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram

        # Generate signals
        df['signal'] = 0
        for i in range(1, len(df)):
            if check_crossover(df['macd'].values, df['macd_signal'].values, i):
                df.loc[df.index[i], 'signal'] = 1  # Buy
            elif check_crossunder(df['macd'].values, df['macd_signal'].values, i):
                df.loc[df.index[i], 'signal'] = -1  # Sell

        return df


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy.

    Buy when price touches lower band (oversold).
    Sell when price touches upper band (overbought).
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands strategy.

        Args:
            period: Period for moving average
            std_dev: Number of standard deviations
        """
        super().__init__(f"BB_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Bollinger Bands."""
        df = data.copy()

        # Calculate Bollinger Bands
        upper, middle, lower = bollinger_bands(df['Close'], self.period, self.std_dev)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower

        # Generate signals
        df['signal'] = 0
        for i in range(1, len(df)):
            # Buy when price crosses above lower band
            if df['Close'].iloc[i - 1] <= df['bb_lower'].iloc[i - 1] and \
               df['Close'].iloc[i] > df['bb_lower'].iloc[i]:
                df.loc[df.index[i], 'signal'] = 1
            # Sell when price crosses below upper band
            elif df['Close'].iloc[i - 1] >= df['bb_upper'].iloc[i - 1] and \
                 df['Close'].iloc[i] < df['bb_upper'].iloc[i]:
                df.loc[df.index[i], 'signal'] = -1

        return df


class CombinedStrategy(BaseStrategy):
    """
    Combined Multi-Indicator Strategy.

    Uses multiple indicators for confirmation:
    - SMA trend direction
    - RSI for momentum
    - MACD for confirmation

    Buy when: SMA uptrend + RSI oversold recovery + MACD bullish
    Sell when: SMA downtrend + RSI overbought decline + MACD bearish
    """

    def __init__(self,
                 sma_fast: int = 13,
                 sma_slow: int = 26,
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9):
        """
        Initialize combined strategy.

        Args:
            sma_fast: Fast SMA period
            sma_slow: Slow SMA period
            rsi_period: RSI period
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
        """
        super().__init__("Combined_Multi_Indicator")
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on multiple indicators."""
        df = data.copy()

        # Calculate all indicators
        df['sma_fast'] = SMA(df['Close'], self.sma_fast)
        df['sma_slow'] = SMA(df['Close'], self.sma_slow)
        df['rsi'] = RSI(df['Close'], self.rsi_period)

        macd_line, signal_line, histogram = MACD(
            df['Close'], self.macd_fast, self.macd_slow, self.macd_signal
        )
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram

        # Generate signals with confirmation
        df['signal'] = 0
        for i in range(1, len(df)):
            # Trend: SMA fast above slow (uptrend)
            sma_uptrend = df['sma_fast'].iloc[i] > df['sma_slow'].iloc[i]
            sma_downtrend = df['sma_fast'].iloc[i] < df['sma_slow'].iloc[i]

            # Momentum: RSI conditions
            rsi_oversold_recovery = df['rsi'].iloc[i - 1] < 40 and df['rsi'].iloc[i] >= 40
            rsi_overbought_decline = df['rsi'].iloc[i - 1] > 60 and df['rsi'].iloc[i] <= 60

            # Confirmation: MACD crossover
            macd_bullish = check_crossover(df['macd'].values, df['macd_signal'].values, i)
            macd_bearish = check_crossunder(df['macd'].values, df['macd_signal'].values, i)

            # Alternative: MACD histogram positive/negative
            macd_positive = df['macd_hist'].iloc[i] > 0
            macd_negative = df['macd_hist'].iloc[i] < 0

            # Buy signal: uptrend + momentum recovery + MACD confirmation
            if (sma_uptrend and rsi_oversold_recovery) or \
               (sma_uptrend and macd_bullish):
                df.loc[df.index[i], 'signal'] = 1

            # Sell signal: downtrend + momentum decline + MACD confirmation
            elif (sma_downtrend and rsi_overbought_decline) or \
                 (sma_downtrend and macd_bearish):
                df.loc[df.index[i], 'signal'] = -1

        return df


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy using ADX.

    Only takes trades when trend is strong (ADX > 25).
    Uses EMA for trend direction and entry signals.
    """

    def __init__(self,
                 ema_fast: int = 9,
                 ema_slow: int = 21,
                 adx_period: int = 14,
                 adx_threshold: float = 25):
        """
        Initialize trend following strategy.

        Args:
            ema_fast: Fast EMA period
            ema_slow: Slow EMA period
            adx_period: ADX period
            adx_threshold: Minimum ADX for strong trend
        """
        super().__init__(f"TrendFollowing_EMA_{ema_fast}_{ema_slow}_ADX_{adx_period}")
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on trend strength and EMA crossover."""
        df = data.copy()

        # Calculate indicators
        df['ema_fast'] = EMA(df['Close'], self.ema_fast)
        df['ema_slow'] = EMA(df['Close'], self.ema_slow)
        df['adx'] = ADX(df['High'], df['Low'], df['Close'], self.adx_period)

        # Generate signals only in strong trends
        df['signal'] = 0
        for i in range(1, len(df)):
            # Check if trend is strong
            strong_trend = df['adx'].iloc[i] > self.adx_threshold

            if strong_trend:
                # Buy on EMA crossover in strong uptrend
                if check_crossover(df['ema_fast'].values, df['ema_slow'].values, i):
                    df.loc[df.index[i], 'signal'] = 1
                # Sell on EMA crossunder in strong downtrend
                elif check_crossunder(df['ema_fast'].values, df['ema_slow'].values, i):
                    df.loc[df.index[i], 'signal'] = -1

        return df


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using RSI and Bollinger Bands.

    Buys oversold conditions, sells overbought conditions.
    Uses both RSI and Bollinger Bands for confirmation.
    """

    def __init__(self,
                 rsi_period: int = 14,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70):
        """
        Initialize mean reversion strategy.

        Args:
            rsi_period: RSI period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviations
            rsi_oversold: RSI oversold level
            rsi_overbought: RSI overbought level
        """
        super().__init__(f"MeanReversion_RSI_{rsi_period}_BB_{bb_period}")
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on mean reversion indicators."""
        df = data.copy()

        # Calculate indicators
        df['rsi'] = RSI(df['Close'], self.rsi_period)
        upper, middle, lower = bollinger_bands(df['Close'], self.bb_period, self.bb_std)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower

        # Calculate distance from bands (%)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Generate signals with dual confirmation
        df['signal'] = 0
        for i in range(1, len(df)):
            # Buy when oversold on both indicators
            rsi_oversold_buy = df['rsi'].iloc[i] < self.rsi_oversold
            bb_oversold_buy = df['bb_position'].iloc[i] < 0.2  # Price near lower band

            # Sell when overbought on both indicators
            rsi_overbought_sell = df['rsi'].iloc[i] > self.rsi_overbought
            bb_overbought_sell = df['bb_position'].iloc[i] > 0.8  # Price near upper band

            # Confirmation: reversal from extreme
            if rsi_oversold_buy and bb_oversold_buy:
                df.loc[df.index[i], 'signal'] = 1

            elif rsi_overbought_sell and bb_overbought_sell:
                df.loc[df.index[i], 'signal'] = -1

        return df
