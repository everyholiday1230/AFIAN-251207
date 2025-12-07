"""
Data Normalization
==================

Convert absolute prices to percentage changes for model training.
This is critical for creating scale-invariant features.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict

from src.utils.logger import get_logger

logger = get_logger("normalizer")


class PriceNormalizer:
    """
    Normalize price data to percentage changes.
    
    Philosophy:
    - Models should learn patterns in relative price movements, not absolute prices
    - BTC at $45,000 vs $90,000 should have similar features if patterns are similar
    - Percentage changes are scale-invariant and stationary
    """
    
    def __init__(self):
        self.reference_prices: Dict[str, float] = {}
    
    def normalize_ohlcv(
        self,
        df: pd.DataFrame,
        reference_col: str = 'close',
        periods: List[int] = [1, 5, 15, 60]
    ) -> pd.DataFrame:
        """
        Normalize OHLCV data to percentage changes.
        
        Args:
            df: DataFrame with OHLCV data
            reference_col: Column to use as reference (default: 'close')
            periods: List of lookback periods for percentage changes
        
        Returns:
            DataFrame with normalized features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for normalization")
            return df
        
        result = df.copy()
        
        # Ensure data is sorted by time
        if 'time' in result.columns:
            result = result.sort_values('time')
        
        # Calculate percentage changes for multiple periods
        for period in periods:
            col_name = f'price_change_{period}'
            result[col_name] = result[reference_col].pct_change(period)
        
        # OHLC normalized relative to close
        if all(col in result.columns for col in ['open', 'high', 'low', 'close']):
            result['open_close_pct'] = (result['open'] - result['close']) / result['close']
            result['high_close_pct'] = (result['high'] - result['close']) / result['close']
            result['low_close_pct'] = (result['low'] - result['close']) / result['close']
            
            # High-Low range as percentage of close
            result['hl_range_pct'] = (result['high'] - result['low']) / result['close']
        
        # Volume changes
        if 'volume' in result.columns:
            result['volume_change'] = result['volume'].pct_change(1)
            result['volume_ma_ratio'] = result['volume'] / result['volume'].rolling(20).mean()
        
        # Log returns (alternative to percentage changes)
        result['log_return'] = np.log(result[reference_col] / result[reference_col].shift(1))
        
        # Replace infinities with NaN
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        logger.debug(f"Normalized {len(result)} records with {len(periods)} periods")
        
        return result
    
    def normalize_orderbook(self, orderbook_data: Dict) -> Dict:
        """
        Normalize orderbook data to percentage-based metrics.
        
        Args:
            orderbook_data: Dictionary with bids, asks, and metrics
        
        Returns:
            Dictionary with normalized orderbook metrics
        """
        if not orderbook_data or 'bids' not in orderbook_data:
            return {}
        
        bids = orderbook_data.get('bids', [])
        asks = orderbook_data.get('asks', [])
        
        if not bids or not asks:
            return {}
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        
        # Normalize spread as percentage of mid price
        spread_pct = (best_ask - best_bid) / mid_price if mid_price > 0 else 0
        
        # Calculate weighted average bid/ask prices
        total_bid_volume = sum([bid[1] for bid in bids])
        total_ask_volume = sum([ask[1] for ask in asks])
        
        weighted_bid = sum([bid[0] * bid[1] for bid in bids]) / total_bid_volume if total_bid_volume > 0 else best_bid
        weighted_ask = sum([ask[0] * ask[1] for ask in asks]) / total_ask_volume if total_ask_volume > 0 else best_ask
        
        # Distance from mid price (normalized)
        bid_distance_pct = (mid_price - weighted_bid) / mid_price
        ask_distance_pct = (weighted_ask - mid_price) / mid_price
        
        return {
            'spread_pct': spread_pct,
            'imbalance': orderbook_data.get('imbalance', 0),
            'bid_distance_pct': bid_distance_pct,
            'ask_distance_pct': ask_distance_pct,
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            'volume_ratio': total_bid_volume / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0.5,
        }
    
    def normalize_trades(self, trades_df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """
        Normalize trade data.
        
        Args:
            trades_df: DataFrame with trade data
            window: Rolling window for calculations
        
        Returns:
            DataFrame with normalized trade metrics
        """
        if trades_df.empty:
            return trades_df
        
        result = trades_df.copy()
        
        # Price changes
        result['price_change'] = result['price'].pct_change()
        
        # Volume-weighted average price (VWAP)
        result['vwap'] = (result['price'] * result['amount']).rolling(window).sum() / result['amount'].rolling(window).sum()
        result['vwap_distance'] = (result['price'] - result['vwap']) / result['vwap']
        
        # Buy/Sell pressure
        if 'is_buy' in result.columns or 'side' in result.columns:
            if 'is_buy' not in result.columns:
                result['is_buy'] = result['side'] == 'buy'
            
            # Rolling buy/sell ratio
            result['buy_volume'] = result['amount'].where(result['is_buy'], 0)
            result['sell_volume'] = result['amount'].where(~result['is_buy'], 0)
            
            result['buy_ratio'] = result['buy_volume'].rolling(window).sum() / result['amount'].rolling(window).sum()
            result['sell_ratio'] = result['sell_volume'].rolling(window).sum() / result['amount'].rolling(window).sum()
        
        # Trade intensity (trades per minute)
        if 'timestamp' in result.columns or 'datetime' in result.columns:
            time_col = 'datetime' if 'datetime' in result.columns else 'timestamp'
            result['trade_intensity'] = result.groupby(pd.Grouper(key=time_col, freq='1min')).size()
        
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return result
    
    def create_multi_timeframe_features(
        self,
        df: pd.DataFrame,
        timeframes: List[str] = ['5m', '15m', '1h']
    ) -> pd.DataFrame:
        """
        Create features from multiple timeframes.
        
        Args:
            df: DataFrame with minute-level data
            timeframes: List of timeframes to aggregate
        
        Returns:
            DataFrame with multi-timeframe features
        """
        if df.empty or 'time' not in df.columns:
            return df
        
        result = df.copy()
        result.set_index('time', inplace=True)
        
        for tf in timeframes:
            # Resample to timeframe
            resampled = result.resample(tf).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Calculate percentage changes
            resampled[f'price_change_{tf}'] = resampled['close'].pct_change()
            resampled[f'volatility_{tf}'] = resampled['close'].pct_change().rolling(20).std()
            
            # Merge back to original timeframe (forward fill)
            for col in [f'price_change_{tf}', f'volatility_{tf}']:
                result = result.join(resampled[[col]], how='left')
                result[col].fillna(method='ffill', inplace=True)
        
        result.reset_index(inplace=True)
        
        return result
    
    def denormalize_prediction(
        self,
        prediction_pct: float,
        current_price: float
    ) -> float:
        """
        Convert percentage prediction back to absolute price.
        
        Args:
            prediction_pct: Predicted percentage change
            current_price: Current price
        
        Returns:
            Predicted absolute price
        """
        return current_price * (1 + prediction_pct)
    
    def calculate_returns(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        periods: List[int] = [1, 5, 15, 60]
    ) -> pd.DataFrame:
        """
        Calculate returns for multiple periods.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for prices
            periods: List of periods for return calculation
        
        Returns:
            DataFrame with return columns
        """
        result = df.copy()
        
        for period in periods:
            # Simple returns
            result[f'return_{period}'] = result[price_col].pct_change(period)
            
            # Log returns
            result[f'log_return_{period}'] = np.log(
                result[price_col] / result[price_col].shift(period)
            )
            
            # Forward returns (for labeling)
            result[f'forward_return_{period}'] = result[price_col].pct_change(period).shift(-period)
        
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return result
    
    def robust_scale(
        self,
        data: np.ndarray,
        quantile_range: tuple = (25, 75)
    ) -> np.ndarray:
        """
        Robust scaling using quantiles (less sensitive to outliers).
        
        Args:
            data: Input data
            quantile_range: Quantile range for scaling
        
        Returns:
            Scaled data
        """
        q1 = np.percentile(data, quantile_range[0])
        q3 = np.percentile(data, quantile_range[1])
        iqr = q3 - q1
        
        if iqr == 0:
            return data - np.median(data)
        
        return (data - np.median(data)) / iqr
    
    def clip_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        Clip outliers beyond n standard deviations.
        
        Args:
            df: Input DataFrame
            columns: Columns to clip
            n_std: Number of standard deviations
        
        Returns:
            DataFrame with clipped values
        """
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            mean = result[col].mean()
            std = result[col].std()
            
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
            
            result[col] = result[col].clip(lower_bound, upper_bound)
        
        return result


# Global normalizer instance
price_normalizer = PriceNormalizer()


if __name__ == "__main__":
    # Test normalization
    print("=== Price Normalization Test ===\n")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    prices = 45000 + np.cumsum(np.random.randn(100) * 100)
    volumes = np.random.rand(100) * 1000
    
    df = pd.DataFrame({
        'time': dates,
        'open': prices + np.random.randn(100) * 50,
        'high': prices + abs(np.random.randn(100) * 100),
        'low': prices - abs(np.random.randn(100) * 100),
        'close': prices,
        'volume': volumes,
    })
    
    print("Original Data:")
    print(df[['time', 'close', 'volume']].head())
    print()
    
    # Normalize
    normalizer = PriceNormalizer()
    normalized = normalizer.normalize_ohlcv(df, periods=[1, 5, 15])
    
    print("Normalized Data (new columns):")
    print(normalized[['time', 'close', 'price_change_1', 'price_change_5', 'volume_change']].head(20))
    print()
    
    # Test orderbook normalization
    orderbook = {
        'bids': [[45000, 1.5], [44990, 2.0], [44980, 1.0]],
        'asks': [[45010, 1.2], [45020, 1.8], [45030, 0.8]],
        'imbalance': 0.15,
    }
    
    normalized_ob = normalizer.normalize_orderbook(orderbook)
    print("Normalized Orderbook:")
    for key, value in normalized_ob.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n✅ Normalization test completed!")
