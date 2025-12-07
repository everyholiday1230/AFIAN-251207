"""
Feature Engineering
===================

Create sophisticated features for model training.

Feature Categories:
1. Technical Indicators (RSI, MACD, Bollinger Bands, etc.)
2. Market Microstructure (Order Book, Trade Flow)
3. Volatility Metrics
4. Futures-Specific (Funding Rate, Open Interest)
5. Multi-Timeframe Features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import ta  # Technical Analysis library
from numba import jit

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger("feature_engineer")


class FeatureEngineer:
    """
    Comprehensive feature engineering for trading models.
    
    All features are percentage-based or normalized for scale invariance.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
    
    # ========================================================================
    # Technical Indicators
    # ========================================================================
    
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Add comprehensive technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for prices
        
        Returns:
            DataFrame with technical indicators
        """
        result = df.copy()
        
        # RSI (Relative Strength Index) - Already normalized 0-100
        result['rsi_14'] = ta.momentum.RSIIndicator(
            close=result[price_col], window=14
        ).rsi() / 100.0  # Normalize to 0-1
        
        result['rsi_7'] = ta.momentum.RSIIndicator(
            close=result[price_col], window=7
        ).rsi() / 100.0
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close=result[price_col])
        result['macd'] = macd.macd_diff() / result[price_col]  # Normalize by price
        result['macd_signal'] = macd.macd_signal() / result[price_col]
        
        # Bollinger Bands - Position within bands
        bollinger = ta.volatility.BollingerBands(
            close=result[price_col], window=20, window_dev=2
        )
        bb_high = bollinger.bollinger_hband()
        bb_low = bollinger.bollinger_lband()
        bb_mid = bollinger.bollinger_mavg()
        
        # BB position: 0 = lower band, 0.5 = middle, 1 = upper band
        result['bb_position'] = (result[price_col] - bb_low) / (bb_high - bb_low)
        result['bb_width'] = (bb_high - bb_low) / bb_mid  # Band width (volatility indicator)
        
        # Moving Averages - Distance from price
        for window in [7, 14, 21, 50]:
            ma = result[price_col].rolling(window).mean()
            result[f'ma_{window}_dist'] = (result[price_col] - ma) / ma
        
        # EMA (Exponential Moving Average)
        for span in [12, 26, 50]:
            ema = result[price_col].ewm(span=span).mean()
            result[f'ema_{span}_dist'] = (result[price_col] - ema) / ema
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=result['high'],
            low=result['low'],
            close=result[price_col],
            window=14,
            smooth_window=3
        )
        result['stoch_k'] = stoch.stoch() / 100.0  # Normalize to 0-1
        result['stoch_d'] = stoch.stoch_signal() / 100.0
        
        # ADX (Average Directional Index) - Trend strength
        adx = ta.trend.ADXIndicator(
            high=result['high'],
            low=result['low'],
            close=result[price_col],
            window=14
        )
        result['adx'] = adx.adx() / 100.0  # Normalize to 0-1
        result['adx_pos'] = adx.adx_pos() / 100.0
        result['adx_neg'] = adx.adx_neg() / 100.0
        
        # CCI (Commodity Channel Index)
        cci = ta.trend.CCIIndicator(
            high=result['high'],
            low=result['low'],
            close=result[price_col],
            window=20
        )
        result['cci'] = cci.cci() / 100.0  # Normalize
        
        # ATR (Average True Range) - Volatility
        atr = ta.volatility.AverageTrueRange(
            high=result['high'],
            low=result['low'],
            close=result[price_col],
            window=14
        )
        result['atr'] = atr.average_true_range() / result[price_col]  # Normalize by price
        
        # OBV (On-Balance Volume) - Volume-price trend
        obv = ta.volume.OnBalanceVolumeIndicator(
            close=result[price_col],
            volume=result['volume']
        )
        result['obv'] = obv.on_balance_volume()
        result['obv_change'] = result['obv'].pct_change()
        
        logger.debug(f"Added {len([c for c in result.columns if c not in df.columns])} technical indicators")
        
        return result
    
    # ========================================================================
    # Volatility Features
    # ========================================================================
    
    def add_volatility_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Add volatility-based features.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for prices
        
        Returns:
            DataFrame with volatility features
        """
        result = df.copy()
        
        # Calculate returns
        returns = result[price_col].pct_change()
        
        # Historical volatility (different windows)
        for window in [10, 20, 50]:
            result[f'volatility_{window}'] = returns.rolling(window).std()
        
        # Parkinson volatility (high-low range based)
        result['parkinson_vol'] = self._calculate_parkinson_volatility(
            result['high'], result['low']
        )
        
        # Garman-Klass volatility (OHLC based)
        result['gk_vol'] = self._calculate_gk_volatility(
            result['open'], result['high'], result['low'], result[price_col]
        )
        
        # Volatility ratio (short-term / long-term)
        result['vol_ratio'] = result['volatility_10'] / result['volatility_50']
        
        # Price range features
        result['high_low_range'] = (result['high'] - result['low']) / result[price_col]
        result['open_close_range'] = abs(result['open'] - result[price_col]) / result[price_col]
        
        # Volatility regime (relative to historical average)
        result['vol_regime'] = result['volatility_20'] / result['volatility_20'].rolling(100).mean()
        
        return result
    
    @staticmethod
    def _calculate_parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        hl_ratio = np.log(high / low)
        return (hl_ratio ** 2 / (4 * np.log(2))).rolling(window).mean().apply(np.sqrt)
    
    @staticmethod
    def _calculate_gk_volatility(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Calculate Garman-Klass volatility estimator."""
        hl = np.log(high / low)
        co = np.log(close / open_)
        
        gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
        return gk.rolling(window).mean().apply(np.sqrt)
    
    # ========================================================================
    # Market Microstructure Features
    # ========================================================================
    
    def add_market_microstructure_features(
        self,
        df: pd.DataFrame,
        orderbook_col: Optional[str] = None,
        trades_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add market microstructure features.
        
        Args:
            df: DataFrame with market data
            orderbook_col: Column with orderbook imbalance
            trades_col: Column with trade data
        
        Returns:
            DataFrame with microstructure features
        """
        result = df.copy()
        
        # Volume-based features
        if 'volume' in result.columns:
            # Volume moving averages
            result['volume_ma_20'] = result['volume'].rolling(20).mean()
            result['volume_ratio'] = result['volume'] / result['volume_ma_20']
            
            # Volume trend
            result['volume_trend'] = result['volume'].rolling(10).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
            )
            
            # Volume acceleration
            result['volume_accel'] = result['volume'].diff().diff()
            
            # VWAP (Volume Weighted Average Price)
            if 'close' in result.columns:
                result['vwap'] = (result['close'] * result['volume']).rolling(20).sum() / result['volume'].rolling(20).sum()
                result['vwap_dist'] = (result['close'] - result['vwap']) / result['vwap']
        
        # Order book imbalance (if available)
        if orderbook_col and orderbook_col in result.columns:
            result['ob_imbalance_ma'] = result[orderbook_col].rolling(20).mean()
            result['ob_imbalance_std'] = result[orderbook_col].rolling(20).std()
            result['ob_imbalance_zscore'] = (
                (result[orderbook_col] - result['ob_imbalance_ma']) / result['ob_imbalance_std']
            )
        
        # Price momentum
        if 'close' in result.columns:
            for period in [5, 10, 20]:
                result[f'momentum_{period}'] = result['close'].pct_change(period)
            
            # Price acceleration
            result['price_accel'] = result['close'].diff().diff()
        
        return result
    
    # ========================================================================
    # Futures-Specific Features
    # ========================================================================
    
    def add_futures_features(
        self,
        df: pd.DataFrame,
        funding_rate_col: Optional[str] = 'funding_rate'
    ) -> pd.DataFrame:
        """
        Add futures-specific features.
        
        Args:
            df: DataFrame with futures data
            funding_rate_col: Column with funding rate
        
        Returns:
            DataFrame with futures features
        """
        result = df.copy()
        
        # Funding rate features (if available)
        if funding_rate_col and funding_rate_col in result.columns:
            # Funding rate moving average
            result['funding_rate_ma'] = result[funding_rate_col].rolling(8).mean()  # 8 periods (daily)
            
            # Funding rate deviation
            result['funding_rate_std'] = result[funding_rate_col].rolling(8).std()
            result['funding_rate_zscore'] = (
                (result[funding_rate_col] - result['funding_rate_ma']) / result['funding_rate_std']
            )
            
            # Funding rate trend
            result['funding_rate_trend'] = result[funding_rate_col].rolling(8).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
            )
            
            # Cumulative funding cost
            result['cumulative_funding'] = result[funding_rate_col].rolling(24).sum()  # Last 24 funding periods
        
        return result
    
    # ========================================================================
    # Time-Based Features
    # ========================================================================
    
    def add_time_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'time'
    ) -> pd.DataFrame:
        """
        Add time-based cyclic features.
        
        Args:
            df: DataFrame with timestamp
            timestamp_col: Column name for timestamps
        
        Returns:
            DataFrame with time features
        """
        result = df.copy()
        
        if timestamp_col not in result.columns:
            return result
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
            result[timestamp_col] = pd.to_datetime(result[timestamp_col])
        
        # Hour of day (cyclic encoding)
        hour = result[timestamp_col].dt.hour
        result['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        result['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (cyclic encoding)
        day = result[timestamp_col].dt.dayofweek
        result['day_sin'] = np.sin(2 * np.pi * day / 7)
        result['day_cos'] = np.cos(2 * np.pi * day / 7)
        
        # Day of month (cyclic encoding)
        day_of_month = result[timestamp_col].dt.day
        result['dom_sin'] = np.sin(2 * np.pi * day_of_month / 31)
        result['dom_cos'] = np.cos(2 * np.pi * day_of_month / 31)
        
        # Trading session (Asian, European, American)
        result['is_asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
        result['is_european_session'] = ((hour >= 8) & (hour < 16)).astype(int)
        result['is_american_session'] = ((hour >= 16) & (hour < 24)).astype(int)
        
        return result
    
    # ========================================================================
    # Lag Features
    # ========================================================================
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Add lagged features.
        
        Args:
            df: DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features
        """
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            for lag in lags:
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)
        
        return result
    
    # ========================================================================
    # Rolling Statistics
    # ========================================================================
    
    def add_rolling_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Add rolling statistics features.
        
        Args:
            df: DataFrame
            columns: Columns to calculate statistics for
            windows: List of window sizes
        
        Returns:
            DataFrame with rolling statistics
        """
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            for window in windows:
                # Rolling mean
                result[f'{col}_ma_{window}'] = result[col].rolling(window).mean()
                
                # Rolling std
                result[f'{col}_std_{window}'] = result[col].rolling(window).std()
                
                # Rolling min/max
                result[f'{col}_min_{window}'] = result[col].rolling(window).min()
                result[f'{col}_max_{window}'] = result[col].rolling(window).max()
                
                # Position in range
                range_val = result[f'{col}_max_{window}'] - result[f'{col}_min_{window}']
                result[f'{col}_position_{window}'] = (
                    (result[col] - result[f'{col}_min_{window}']) / range_val
                ).fillna(0.5)
        
        return result
    
    # ========================================================================
    # Master Feature Engineering Pipeline
    # ========================================================================
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        include_technical: bool = True,
        include_volatility: bool = True,
        include_microstructure: bool = True,
        include_futures: bool = True,
        include_time: bool = True
    ) -> pd.DataFrame:
        """
        Create all features in one pipeline.
        
        Args:
            df: DataFrame with raw OHLCV data
            include_technical: Include technical indicators
            include_volatility: Include volatility features
            include_microstructure: Include market microstructure
            include_futures: Include futures-specific features
            include_time: Include time-based features
        
        Returns:
            DataFrame with all features
        """
        logger.info("Starting comprehensive feature engineering...")
        
        result = df.copy()
        initial_cols = len(result.columns)
        
        # Technical indicators
        if include_technical:
            logger.debug("Adding technical indicators...")
            result = self.add_technical_indicators(result)
        
        # Volatility features
        if include_volatility:
            logger.debug("Adding volatility features...")
            result = self.add_volatility_features(result)
        
        # Market microstructure
        if include_microstructure:
            logger.debug("Adding market microstructure features...")
            result = self.add_market_microstructure_features(result)
        
        # Futures-specific
        if include_futures:
            logger.debug("Adding futures features...")
            result = self.add_futures_features(result)
        
        # Time-based features
        if include_time:
            logger.debug("Adding time features...")
            result = self.add_time_features(result)
        
        # Remove infinities and NaNs
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        final_cols = len(result.columns)
        features_added = final_cols - initial_cols
        
        logger.info(f"✅ Feature engineering complete. Added {features_added} features.")
        
        # Store feature names
        self.feature_names = [col for col in result.columns if col not in df.columns]
        
        return result
    
    def get_feature_importance_ready(
        self,
        df: pd.DataFrame,
        target_col: str = 'tb_label'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Args:
            df: DataFrame with features and labels
            target_col: Target column name
        
        Returns:
            Tuple of (features_df, target_series)
        """
        # Exclude non-feature columns
        exclude_cols = [
            'time', 'timestamp', 'symbol', 'timeframe',
            target_col, 'tb_exit_price', 'tb_exit_time', 'tb_return',
            'tb_holding_minutes', 'tb_exit_reason',
            'open', 'high', 'low', 'close', 'volume'  # Raw OHLCV
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with NaN in features or target
        clean_df = df[feature_cols + [target_col]].dropna()
        
        X = clean_df[feature_cols]
        y = clean_df[target_col]
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        
        return X, y


# Global feature engineer instance
feature_engineer = FeatureEngineer()


if __name__ == "__main__":
    # Test feature engineering
    print("=== Feature Engineering Test ===\n")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    prices = 45000 + np.cumsum(np.random.randn(200) * 100)
    
    df = pd.DataFrame({
        'time': dates,
        'open': prices + np.random.randn(200) * 50,
        'high': prices + abs(np.random.randn(200) * 100),
        'low': prices - abs(np.random.randn(200) * 100),
        'close': prices,
        'volume': np.random.rand(200) * 1000 + 500,
        'funding_rate': np.random.randn(200) * 0.0001,
    })
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Create all features
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    print(f"After feature engineering: {df_features.shape}")
    print(f"Features added: {len(df_features.columns) - len(df.columns)}\n")
    
    # Show some features
    feature_cols = [col for col in df_features.columns if col not in df.columns]
    print(f"Sample of new features ({len(feature_cols)} total):")
    for i, col in enumerate(feature_cols[:20]):
        print(f"  {i+1}. {col}")
    
    print("\n✅ Feature engineering test completed!")
