"""
Layer 2: Market Regime Detection
=================================

Detect market conditions to adjust trading strategy:
- Trend: Bull, Bear, Sideways
- Volatility: High, Medium, Low
- Volume: High, Medium, Low

Usage:
    detector = RegimeDetector()
    regime = detector.detect_regime(df)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("regime_detector")


class TrendRegime(Enum):
    """Trend direction."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


class VolatilityRegime(Enum):
    """Volatility level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class VolumeRegime(Enum):
    """Volume level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RegimeDetector:
    """
    Detect market regime using multiple indicators.
    
    Methods:
    - Trend Detection: SMA crossover, ADX
    - Volatility: ATR, Bollinger Band width
    - Volume: Relative volume
    """
    
    def __init__(
        self,
        trend_window: int = 50,
        volatility_window: int = 20,
        volume_window: int = 20
    ):
        """
        Initialize regime detector.
        
        Args:
            trend_window: Window for trend detection
            volatility_window: Window for volatility calculation
            volume_window: Window for volume analysis
        """
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        
        logger.info(f"RegimeDetector initialized")
        logger.info(f"  Trend window: {trend_window}")
        logger.info(f"  Volatility window: {volatility_window}")
        logger.info(f"  Volume window: {volume_window}")
    
    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime for each candle.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with regime columns added
        """
        df = df.copy()
        
        # Detect trend
        df = self._detect_trend(df)
        
        # Detect volatility
        df = self._detect_volatility(df)
        
        # Detect volume
        df = self._detect_volume(df)
        
        # Combined regime
        df['regime'] = df.apply(
            lambda row: f"{row['trend_regime']}_{row['volatility_regime']}_{row['volume_regime']}",
            axis=1
        )
        
        logger.info("✅ Regime detection completed")
        
        return df
    
    def _detect_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect trend regime using SMA and ADX."""
        # Calculate SMAs
        df['sma_short'] = df['close'].rolling(window=20).mean()
        df['sma_long'] = df['close'].rolling(window=self.trend_window).mean()
        
        # Calculate ADX for trend strength
        df['adx'] = self._calculate_adx(df, period=14)
        
        # Determine trend
        def get_trend(row):
            if pd.isna(row['sma_short']) or pd.isna(row['sma_long']) or pd.isna(row['adx']):
                return TrendRegime.SIDEWAYS.value
            
            # Strong trend (ADX > 25)
            if row['adx'] > 25:
                if row['sma_short'] > row['sma_long']:
                    return TrendRegime.BULL.value
                else:
                    return TrendRegime.BEAR.value
            else:
                # Weak trend = Sideways
                return TrendRegime.SIDEWAYS.value
        
        df['trend_regime'] = df.apply(get_trend, axis=1)
        
        return df
    
    def _detect_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect volatility regime using ATR and BB width."""
        # Calculate ATR
        df['atr'] = self._calculate_atr(df, period=self.volatility_window)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Calculate Bollinger Band width
        df['bb_middle'] = df['close'].rolling(window=self.volatility_window).mean()
        df['bb_std'] = df['close'].rolling(window=self.volatility_window).std()
        df['bb_width'] = (df['bb_std'] * 2) / df['bb_middle'] * 100
        
        # Classify volatility (using percentiles)
        if len(df) > 100:
            atr_high = df['atr_pct'].quantile(0.75)
            atr_low = df['atr_pct'].quantile(0.25)
            
            def get_volatility(row):
                if pd.isna(row['atr_pct']):
                    return VolatilityRegime.MEDIUM.value
                
                if row['atr_pct'] > atr_high:
                    return VolatilityRegime.HIGH.value
                elif row['atr_pct'] < atr_low:
                    return VolatilityRegime.LOW.value
                else:
                    return VolatilityRegime.MEDIUM.value
            
            df['volatility_regime'] = df.apply(get_volatility, axis=1)
        else:
            df['volatility_regime'] = VolatilityRegime.MEDIUM.value
        
        return df
    
    def _detect_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect volume regime."""
        # Calculate relative volume
        df['volume_sma'] = df['volume'].rolling(window=self.volume_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Classify volume
        def get_volume(row):
            if pd.isna(row['volume_ratio']):
                return VolumeRegime.MEDIUM.value
            
            if row['volume_ratio'] > 1.5:
                return VolumeRegime.HIGH.value
            elif row['volume_ratio'] < 0.5:
                return VolumeRegime.LOW.value
            else:
                return VolumeRegime.MEDIUM.value
        
        df['volume_regime'] = df.apply(get_volume, axis=1)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate ATR
        atr = self._calculate_atr(df, period)
        
        # Calculate directional indicators
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def get_regime_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about regime distribution."""
        stats = {
            'trend': df['trend_regime'].value_counts().to_dict(),
            'volatility': df['volatility_regime'].value_counts().to_dict(),
            'volume': df['volume_regime'].value_counts().to_dict(),
            'combined': df['regime'].value_counts().head(10).to_dict()
        }
        
        return stats


if __name__ == "__main__":
    # Test regime detector
    print("=== Regime Detector Test ===\n")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='15min')
    
    # Simulate price with different regimes
    prices = []
    for i in range(1000):
        if i < 300:  # Bull trend
            base = 45000 + i * 10
        elif i < 600:  # Sideways
            base = 48000 + np.random.randn() * 500
        else:  # Bear trend
            base = 48000 - (i - 600) * 8
        
        prices.append(base + np.random.randn() * 200)
    
    prices = np.array(prices)
    
    df = pd.DataFrame({
        'time': dates,
        'open': prices + np.random.randn(1000) * 50,
        'high': prices + abs(np.random.randn(1000) * 100),
        'low': prices - abs(np.random.randn(1000) * 100),
        'close': prices,
        'volume': np.random.rand(1000) * 100 + 50,
    })
    
    # Detect regime
    detector = RegimeDetector()
    df_with_regime = detector.detect_regime(df)
    
    # Print stats
    stats = detector.get_regime_stats(df_with_regime)
    
    print("\n📊 Regime Statistics:")
    print(f"\nTrend Distribution:")
    for regime, count in stats['trend'].items():
        print(f"  {regime}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nVolatility Distribution:")
    for regime, count in stats['volatility'].items():
        print(f"  {regime}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nVolume Distribution:")
    for regime, count in stats['volume'].items():
        print(f"  {regime}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\n✅ Regime detector test completed!")
