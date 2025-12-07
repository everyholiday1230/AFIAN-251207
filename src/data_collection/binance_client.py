"""
Binance API Client
==================

Unified client for Binance Futures API using CCXT.
Handles both REST API and WebSocket connections.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import ccxt
import pandas as pd
from ccxt.base.errors import (
    BadSymbol,
    ExchangeError,
    NetworkError,
    RateLimitExceeded,
)

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("binance_client")


class BinanceClient:
    """
    Binance Futures API Client.
    
    Features:
    - Automatic rate limit handling
    - Retry logic for network errors
    - Support for both testnet and mainnet
    - OHLCV, funding rate, orderbook, trades data collection
    """
    
    def __init__(self, testnet: bool = None):
        """
        Initialize Binance client.
        
        Args:
            testnet: Use testnet if True. If None, uses config setting.
        """
        self.testnet = testnet if testnet is not None else config.binance.binance_testnet
        self.exchange: Optional[ccxt.binance] = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize CCXT exchange instance."""
        try:
            # Get API credentials
            api_key = config.binance.api_key
            api_secret = config.binance.api_secret
            
            # Initialize exchange
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'rateLimit': 50,  # 50ms between requests
                'options': {
                    'defaultType': 'future',  # Use futures market
                    'adjustForTimeDifference': True,
                },
            })
            
            # Set testnet if enabled
            if self.testnet:
                self.exchange.set_sandbox_mode(True)
                logger.info("🧪 Binance client initialized in TESTNET mode")
            else:
                logger.info("🚀 Binance client initialized in MAINNET mode")
            
            # Load markets
            self.exchange.load_markets()
            logger.info(f"✅ Loaded {len(self.exchange.markets)} markets")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Binance client: {e}")
            raise
    
    def _retry_on_error(self, func, *args, max_retries=3, **kwargs):
        """
        Retry function call on network errors.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            *args, **kwargs: Function arguments
        
        Returns:
            Function result
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitExceeded as e:
                wait_time = 60  # Wait 1 minute for rate limit
                logger.warning(f"Rate limit exceeded, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            except NetworkError as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Network error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(wait_time)
            except ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                raise
        
        raise Exception(f"Failed after {max_retries} retries")
    
    # ========================================================================
    # OHLCV Data Collection
    # ========================================================================
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1m',
        since: Optional[int] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch (max 1500)
        
        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        try:
            ohlcv = self._retry_on_error(
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe,
                since,
                limit
            )
            
            if not ohlcv:
                logger.warning(f"No OHLCV data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol.replace('/', '')
            df['timeframe'] = timeframe
            
            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a date range.
        
        Args:
            symbol: Trading pair
            timeframe: Candlestick timeframe
            start_date: Start date
            end_date: End date (defaults to now)
        
        Returns:
            DataFrame with historical OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Convert to milliseconds
        since_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        # Calculate timeframe duration in ms
        timeframe_duration_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        
        duration = timeframe_duration_ms.get(timeframe, 60 * 1000)
        
        all_data = []
        current_since = since_ms
        
        logger.info(f"Fetching historical data for {symbol} {timeframe} from {start_date} to {end_date}")
        
        while current_since < end_ms:
            df = self.fetch_ohlcv(symbol, timeframe, current_since, limit=1500)
            
            if df.empty:
                break
            
            all_data.append(df)
            
            # Move to next batch
            current_since = int(df['timestamp'].iloc[-1]) + duration
            
            # Progress log
            progress_date = datetime.fromtimestamp(current_since / 1000)
            logger.debug(f"Progress: {progress_date}")
            
            # Rate limiting
            time.sleep(0.1)
        
        if not all_data:
            logger.warning(f"No historical data collected for {symbol} {timeframe}")
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        logger.info(f"✅ Collected {len(result)} candles for {symbol} {timeframe}")
        
        return result
    
    # ========================================================================
    # Funding Rate Data
    # ========================================================================
    
    def fetch_funding_rate(self, symbol: str) -> Dict:
        """
        Fetch current funding rate.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
        
        Returns:
            Dictionary with funding rate data
        """
        try:
            # Convert symbol format
            binance_symbol = symbol.replace('/', '')
            
            ticker = self._retry_on_error(
                self.exchange.fapiPublicGetPremiumIndex,
                {'symbol': binance_symbol}
            )
            
            return {
                'symbol': symbol,
                'funding_rate': float(ticker['lastFundingRate']),
                'mark_price': float(ticker['markPrice']),
                'index_price': float(ticker['indexPrice']),
                'next_funding_time': int(ticker['nextFundingTime']),
                'time': datetime.now(),
            }
        
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return {}
    
    def fetch_funding_rate_history(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.
        
        Args:
            symbol: Trading pair
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of records (max 1000)
        
        Returns:
            DataFrame with historical funding rates
        """
        try:
            binance_symbol = symbol.replace('/', '')
            
            params = {'symbol': binance_symbol, 'limit': limit}
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            history = self._retry_on_error(
                self.exchange.fapiPublicGetFundingRate,
                params
            )
            
            if not history:
                return pd.DataFrame()
            
            df = pd.DataFrame(history)
            df['fundingRate'] = df['fundingRate'].astype(float)
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['symbol'] = symbol
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching funding rate history for {symbol}: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # Order Book Data
    # ========================================================================
    
    def fetch_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """
        Fetch order book snapshot.
        
        Args:
            symbol: Trading pair
            limit: Depth level (5, 10, 20, 50, 100, 500, 1000)
        
        Returns:
            Dictionary with bids, asks, and derived metrics
        """
        try:
            orderbook = self._retry_on_error(
                self.exchange.fetch_order_book,
                symbol,
                limit
            )
            
            bids = orderbook['bids'][:limit]
            asks = orderbook['asks'][:limit]
            
            # Calculate metrics
            bid_volume = sum([bid[1] for bid in bids])
            ask_volume = sum([ask[1] for ask in asks])
            
            # Spread
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            
            # Imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            return {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'spread': spread,
                'imbalance': imbalance,
                'time': datetime.now(),
            }
        
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {}
    
    # ========================================================================
    # Recent Trades
    # ========================================================================
    
    def fetch_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch recent trades.
        
        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)
        
        Returns:
            DataFrame with recent trades
        """
        try:
            trades = self._retry_on_error(
                self.exchange.fetch_trades,
                symbol,
                limit=limit
            )
            
            if not trades:
                return pd.DataFrame()
            
            df = pd.DataFrame(trades)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol.replace('/', '')
            
            # Add derived metrics
            df['is_buy'] = df['side'] == 'buy'
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # Account & Position Data (for live trading)
    # ========================================================================
    
    def fetch_balance(self) -> Dict:
        """Fetch account balance."""
        try:
            balance = self._retry_on_error(self.exchange.fetch_balance)
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}
    
    def fetch_positions(self) -> List[Dict]:
        """Fetch open positions."""
        try:
            positions = self._retry_on_error(
                self.exchange.fapiPrivateGetPositionRisk
            )
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """Get exchange information."""
        try:
            if symbol:
                return self.exchange.market(symbol)
            else:
                return self.exchange.load_markets()
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            return {}
    
    def close(self):
        """Close exchange connection."""
        if self.exchange:
            self.exchange.close()
            logger.info("Binance client closed")


if __name__ == "__main__":
    # Test Binance client
    print("=== Binance Client Test ===\n")
    
    # Initialize client in testnet mode
    print("1. Initializing Binance client (testnet)...")
    client = BinanceClient(testnet=True)
    print("✅ Client initialized\n")
    
    # Test OHLCV fetch
    print("2. Fetching OHLCV data...")
    df = client.fetch_ohlcv('BTC/USDT', '1h', limit=10)
    if not df.empty:
        print(f"✅ Fetched {len(df)} candles")
        print(df[['time', 'open', 'high', 'low', 'close', 'volume']].head())
    else:
        print("❌ Failed to fetch OHLCV")
    print()
    
    # Test funding rate
    print("3. Fetching funding rate...")
    funding = client.fetch_funding_rate('BTC/USDT')
    if funding:
        print(f"✅ Funding Rate: {funding['funding_rate']:.6%}")
        print(f"   Mark Price: ${funding['mark_price']:,.2f}")
    else:
        print("❌ Failed to fetch funding rate")
    print()
    
    # Test order book
    print("4. Fetching order book...")
    orderbook = client.fetch_order_book('BTC/USDT', limit=5)
    if orderbook:
        print(f"✅ Order Book Imbalance: {orderbook['imbalance']:.4f}")
        print(f"   Spread: ${orderbook['spread']:.2f}")
    else:
        print("❌ Failed to fetch order book")
    print()
    
    # Test recent trades
    print("5. Fetching recent trades...")
    trades = client.fetch_trades('BTC/USDT', limit=10)
    if not trades.empty:
        print(f"✅ Fetched {len(trades)} trades")
        buy_ratio = trades['is_buy'].mean()
        print(f"   Buy/Sell Ratio: {buy_ratio:.2%} buys")
    else:
        print("❌ Failed to fetch trades")
    
    print("\n✅ Binance client test completed!")
