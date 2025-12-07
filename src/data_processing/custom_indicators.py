"""
Custom Trading Indicators
==========================

Implements the exact indicators from user's TradingView scripts:
1. AI Learning Optimized Multi-Oscillator (12 features)
2. Impulse MACD [LazyBear]

These are the ONLY indicators used for model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.utils.logger import get_logger

logger = get_logger("custom_indicators")


class CustomIndicators:
    """
    Implements user-specified indicators with exact parameters.
    
    Features (12 total):
    - F1: UPRSI (Dynamic normalized RSI)
    - F2: UPStoch (Dynamic normalized Stochastic)
    - F3: scaled_rsi (Absolute RSI)
    - F4: scaled_mfi (Money Flow Index)
    - F5: rsi_mfi_divergence
    - F6: momentum_balance
    - F7: relative_absolute_diff
    - F8: uprsi_velocity
    - F9: scaled_rsi_velocity
    - F10: avg_acceleration
    - F11: avg_volatility
    - F12: trend_strength
    
    Plus Impulse MACD indicators.
    """
    
    def __init__(
        self,
        # AI Oscillator parameters
        rsi_period: int = 60,
        rsi_lookback: int = 300,
        stoch_length: int = 60,
        smooth_k: int = 9,
        smooth_d: int = 5,
        stoch_lookback: int = 240,
        mfi_length: int = 60,
        velocity_period: int = 1,
        volatility_period: int = 20,
        trend_period: int = 10,
        # Impulse MACD parameters
        length_ma: int = 34,
        length_signal: int = 9
    ):
        """Initialize with exact TradingView parameters."""
        # AI Oscillator
        self.rsi_period = rsi_period
        self.rsi_lookback = rsi_lookback
        self.stoch_length = stoch_length
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d
        self.stoch_lookback = stoch_lookback
        self.mfi_length = mfi_length
        self.velocity_period = velocity_period
        self.volatility_period = volatility_period
        self.trend_period = trend_period
        
        # Impulse MACD
        self.length_ma = length_ma
        self.length_signal = length_signal
        
        logger.info("Custom indicators initialized with exact TradingView parameters")
    
    # ========================================================================
    # Helper Functions
    # ========================================================================
    
    @staticmethod
    def calc_rsi(series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calc_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                       length: int) -> pd.Series:
        """Calculate Stochastic %K."""
        lowest_low = low.rolling(window=length).min()
        highest_high = high.rolling(window=length).max()
        stoch = 100 * (close - lowest_low) / (highest_high - lowest_low)
        return stoch
    
    @staticmethod
    def calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, length: int) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0).rolling(window=length).sum()
        negative_flow = money_flow.where(delta < 0, 0).rolling(window=length).sum()
        
        # MFI
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi
    
    @staticmethod
    def calc_smma(series: pd.Series, length: int) -> pd.Series:
        """Calculate Smoothed Moving Average (like TradingView)."""
        # SMMA = (SMMA[1] * (length - 1) + current) / length
        smma = pd.Series(index=series.index, dtype=float)
        smma.iloc[length-1] = series.iloc[:length].mean()  # Initial SMA
        
        for i in range(length, len(series)):
            smma.iloc[i] = (smma.iloc[i-1] * (length - 1) + series.iloc[i]) / length
        
        return smma
    
    @staticmethod
    def calc_zlema(series: pd.Series, length: int) -> pd.Series:
        """Calculate Zero-Lag EMA."""
        ema1 = series.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        zlema = ema1 + (ema1 - ema2)
        return zlema
    
    # ========================================================================
    # AI Oscillator Features (12 features)
    # ========================================================================
    
    def calculate_ai_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 12 AI Oscillator features.
        
        Args:
            df: DataFrame with OHLCV columns
        
        Returns:
            DataFrame with 12 additional feature columns
        """
        result = df.copy()
        
        # Extract price data
        close = result['close']
        high = result['high']
        low = result['low']
        volume = result['volume']
        
        logger.debug("Calculating AI Oscillator features...")
        
        # ============================================================
        # Core Features (4)
        # ============================================================
        
        # F1: UPRSI - Dynamic normalized RSI
        rsi_value = self.calc_rsi(close, self.rsi_period)
        lowest_rsi = rsi_value.rolling(window=self.rsi_lookback).min()
        highest_rsi = rsi_value.rolling(window=self.rsi_lookback).max()
        range_rsi = highest_rsi - lowest_rsi
        norm_rsi = (rsi_value - lowest_rsi) / range_rsi
        norm_rsi = norm_rsi.fillna(0.5)  # Handle division by zero
        result['F1_UPRSI'] = norm_rsi - 0.5
        
        # F2: UPStoch - Dynamic normalized Stochastic
        stoch_raw = self.calc_stochastic(high, low, close, self.stoch_length)
        k = stoch_raw.rolling(window=self.smooth_k).mean()
        d = k.rolling(window=self.smooth_d).mean()
        lowest_k = k.rolling(window=self.stoch_lookback).min()
        highest_k = k.rolling(window=self.stoch_lookback).max()
        range_k = highest_k - lowest_k
        norm_k = (k - lowest_k) / range_k
        norm_k = norm_k.fillna(0.5)
        result['F2_UPStoch'] = norm_k - 0.5
        
        # F3: scaled_rsi - Absolute RSI
        result['F3_scaled_rsi'] = (rsi_value / 100) - 0.5
        
        # F4: scaled_mfi - Money Flow Index
        mfi_value = self.calc_mfi(high, low, close, volume, self.mfi_length)
        result['F4_scaled_mfi'] = (mfi_value / 100) - 0.5
        
        # ============================================================
        # Derived Features (8)
        # ============================================================
        
        # F5: RSI-MFI Divergence
        result['F5_rsi_mfi_divergence'] = np.abs(
            result['F3_scaled_rsi'] - result['F4_scaled_mfi']
        )
        
        # F6: Momentum Balance
        result['F6_momentum_balance'] = (
            result['F1_UPRSI'] + result['F2_UPStoch'] + 
            result['F3_scaled_rsi'] + result['F4_scaled_mfi']
        ) / 4
        
        # F7: Relative-Absolute Difference
        result['F7_relative_absolute_diff'] = (
            (result['F1_UPRSI'] + result['F2_UPStoch']) / 2 -
            (result['F3_scaled_rsi'] + result['F4_scaled_mfi']) / 2
        )
        
        # F8: UPRSI Velocity (1st derivative)
        result['F8_uprsi_velocity'] = result['F1_UPRSI'].diff(self.velocity_period)
        
        # F9: scaled_rsi Velocity
        result['F9_scaled_rsi_velocity'] = result['F3_scaled_rsi'].diff(self.velocity_period)
        
        # F10: Average Acceleration (2nd derivative)
        uprsi_accel = result['F8_uprsi_velocity'].diff(self.velocity_period)
        scaled_rsi_accel = result['F9_scaled_rsi_velocity'].diff(self.velocity_period)
        result['F10_avg_acceleration'] = (uprsi_accel + scaled_rsi_accel) / 2
        
        # F11: Average Volatility
        uprsi_vol = result['F1_UPRSI'].rolling(window=self.volatility_period).std()
        scaled_rsi_vol = result['F3_scaled_rsi'].rolling(window=self.volatility_period).std()
        result['F11_avg_volatility'] = (uprsi_vol + scaled_rsi_vol) / 2
        
        # F12: Trend Strength (-1 to 1)
        uprsi_direction = (result['F1_UPRSI'] > result['F1_UPRSI'].shift(1)).astype(int) * 2 - 1
        scaled_rsi_direction = (result['F3_scaled_rsi'] > result['F3_scaled_rsi'].shift(1)).astype(int) * 2 - 1
        result['F12_trend_strength'] = (
            uprsi_direction.rolling(window=self.trend_period).sum() +
            scaled_rsi_direction.rolling(window=self.trend_period).sum()
        ) / (2 * self.trend_period)
        
        logger.debug("✅ AI Oscillator features calculated (12 features)")
        
        return result
    
    # ========================================================================
    # Impulse MACD [LazyBear]
    # ========================================================================
    
    def calculate_impulse_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Impulse MACD indicators.
        
        Args:
            df: DataFrame with OHLCV columns
        
        Returns:
            DataFrame with Impulse MACD columns
        """
        result = df.copy()
        
        # HLC3 source
        src = (result['high'] + result['low'] + result['close']) / 3
        
        # SMMA of high and low
        hi = self.calc_smma(result['high'], self.length_ma)
        lo = self.calc_smma(result['low'], self.length_ma)
        
        # ZLEMA of source
        mi = self.calc_zlema(src, self.length_ma)
        
        # Impulse calculation
        md = pd.Series(index=result.index, dtype=float)
        for i in range(len(result)):
            if mi.iloc[i] > hi.iloc[i]:
                md.iloc[i] = mi.iloc[i] - hi.iloc[i]
            elif mi.iloc[i] < lo.iloc[i]:
                md.iloc[i] = mi.iloc[i] - lo.iloc[i]
            else:
                md.iloc[i] = 0
        
        # Signal line
        sb = md.rolling(window=self.length_signal).mean()
        
        # Histogram
        sh = md - sb
        
        result['impulse_macd'] = md
        result['impulse_signal'] = sb
        result['impulse_histogram'] = sh
        
        # Color code (for reference)
        result['impulse_color'] = 0  # Will be mapped later if needed
        
        logger.debug("✅ Impulse MACD calculated (3 indicators)")
        
        return result
    
    # ========================================================================
    # Master Function
    # ========================================================================
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ALL custom indicators.
        
        Args:
            df: DataFrame with OHLCV columns
        
        Returns:
            DataFrame with all 15 indicator columns
        """
        logger.info("Calculating all custom indicators...")
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain {required_cols}")
        
        # Calculate AI Oscillator (12 features)
        result = self.calculate_ai_oscillator(df)
        
        # Calculate Impulse MACD (3 indicators)
        result = self.calculate_impulse_macd(result)
        
        # Drop NaN rows (from rolling calculations)
        initial_rows = len(result)
        result = result.dropna()
        dropped_rows = initial_rows - len(result)
        
        logger.info(f"✅ All indicators calculated")
        logger.info(f"   - AI Oscillator: 12 features")
        logger.info(f"   - Impulse MACD: 3 indicators")
        logger.info(f"   - Total: 15 indicators")
        logger.info(f"   - Rows dropped (NaN): {dropped_rows}")
        logger.info(f"   - Valid rows: {len(result)}")
        
        return result
    
    def get_feature_names(self) -> list:
        """Get list of all feature names."""
        return [
            # AI Oscillator (12)
            'F1_UPRSI',
            'F2_UPStoch',
            'F3_scaled_rsi',
            'F4_scaled_mfi',
            'F5_rsi_mfi_divergence',
            'F6_momentum_balance',
            'F7_relative_absolute_diff',
            'F8_uprsi_velocity',
            'F9_scaled_rsi_velocity',
            'F10_avg_acceleration',
            'F11_avg_volatility',
            'F12_trend_strength',
            # Impulse MACD (3)
            'impulse_macd',
            'impulse_signal',
            'impulse_histogram'
        ]


# Global instance
custom_indicators = CustomIndicators()


if __name__ == "__main__":
    # Test custom indicators
    print("=== Custom Indicators Test ===\n")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='15min')
    
    # Simulate realistic price data
    base_price = 45000
    returns = np.random.randn(500) * 0.002
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'time': dates,
        'open': prices + np.random.randn(500) * 20,
        'high': prices + abs(np.random.randn(500) * 100),
        'low': prices - abs(np.random.randn(500) * 100),
        'close': prices,
        'volume': np.random.rand(500) * 1000 + 500,
    })
    
    print(f"Sample data shape: {df.shape}\n")
    
    # Calculate indicators
    indicators = CustomIndicators()
    df_with_indicators = indicators.calculate_all_indicators(df)
    
    print(f"Data with indicators shape: {df_with_indicators.shape}\n")
    
    # Show feature names
    feature_names = indicators.get_feature_names()
    print(f"Feature names ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i}. {name}")
    
    # Show last 5 rows of features
    print("\nLast 5 rows of features:")
    print(df_with_indicators[feature_names].tail())
    
    # Check for NaN values
    nan_counts = df_with_indicators[feature_names].isna().sum()
    print(f"\nNaN counts per feature:")
    print(nan_counts[nan_counts > 0])
    
    print("\n✅ Custom indicators test completed!")
