"""
Download historical Binance data from public sources

Now supports REAL Binance data download via their public API!
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from tqdm import tqdm
import time
import argparse
from src.utils.logger import get_logger

logger = get_logger("data_download")


def download_binance_klines(symbol: str, interval: str, start_date: str, end_date: str):
    """
    Download real historical kline data from Binance public API.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '15m', '1h', '1d')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        pandas DataFrame with OHLCV data
    """
    logger.info(f"🔄 Downloading REAL Binance data for {symbol} {interval}")
    logger.info(f"   Date range: {start_date} to {end_date}")
    
    # Binance API endpoint
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert dates to timestamps
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    
    # Binance limit is 1000 candles per request
    limit = 1000
    
    with tqdm(desc=f"Downloading {symbol}") as pbar:
        while current_ts < end_ts:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_ts,
                'endTime': end_ts,
                'limit': limit
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # Update progress
                pbar.update(len(data))
                
                # Move to next batch
                current_ts = data[-1][0] + 1
                
                # Rate limiting (to be nice to Binance API)
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading data: {e}")
                break
    
    if not all_data:
        logger.warning("No data downloaded!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert data types
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # Select and reorder columns
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['time']).reset_index(drop=True)
    
    logger.info(f"✅ Downloaded {len(df):,} candles")
    logger.info(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    logger.info(f"   Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    
    return df


def generate_realistic_btc_data(start_date: str, end_date: str, timeframe: str = '15m'):
    """
    Generate realistic BTC/USDT data based on historical patterns
    Uses actual BTC price ranges for each year
    """
    logger.info(f"Generating realistic BTC data from {start_date} to {end_date}")
    
    # Historical BTC price ranges (approximate)
    year_prices = {
        2019: (3500, 13000),    # Bear to bull
        2020: (5000, 29000),    # Bull run start
        2021: (29000, 69000),   # ATH
        2022: (16000, 48000),   # Bear market
        2023: (16500, 45000),   # Recovery
        2024: (40000, 73000),   # New cycle
    }
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Determine frequency
    if timeframe == '15m':
        freq = '15min'
    elif timeframe == '1h':
        freq = '1H'
    elif timeframe == '5m':
        freq = '5min'
    else:
        freq = '15min'
    
    timestamps = pd.date_range(start=start, end=end, freq=freq)
    n = len(timestamps)
    
    logger.info(f"Generating {n} candles...")
    
    # Initialize price array
    np.random.seed(42)  # Reproducible
    prices = []
    
    # Generate prices year by year for more realism
    for ts in tqdm(timestamps, desc="Generating price data"):
        year = ts.year
        if year in year_prices:
            price_min, price_max = year_prices[year]
        else:
            price_min, price_max = (40000, 70000)  # Default
        
        # Random walk within year range
        if not prices:
            base = (price_min + price_max) / 2
        else:
            base = prices[-1]
        
        # Add trend + volatility + mean reversion
        trend = np.random.randn() * 100
        volatility = np.random.randn() * 200
        mean_reversion = ((price_min + price_max) / 2 - base) * 0.001
        
        new_price = base + trend + volatility + mean_reversion
        new_price = np.clip(new_price, price_min, price_max)
        prices.append(new_price)
    
    close = np.array(prices)
    
    # Generate OHLCV
    high = close * (1 + np.abs(np.random.randn(n)) * 0.003)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.003)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    volume = np.random.uniform(50, 500, n)  # BTC volume
    
    df = pd.DataFrame({
        'time': timestamps,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    logger.info(f"✅ Generated {len(df)} candles")
    logger.info(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    logger.info(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    
    return df


def main():
    """Download and save historical data"""
    
    parser = argparse.ArgumentParser(description='Download Binance historical data')
    parser.add_argument('--real', action='store_true', help='Download REAL data from Binance API')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--interval', type=str, default='15m', help='Timeframe (15m, 1h, etc.)')
    parser.add_argument('--start', type=str, default='2019-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.real:
        # Download REAL Binance data
        logger.info("\n" + "="*60)
        logger.info("🚀 DOWNLOADING REAL BINANCE DATA")
        logger.info("="*60)
        
        try:
            df = download_binance_klines(
                symbol=args.symbol,
                interval=args.interval,
                start_date=args.start,
                end_date=args.end
            )
            
            if df is not None:
                # Save combined file
                output_file = output_dir / f"{args.symbol}_{args.interval}_real_{args.start}_{args.end}.csv"
                df.to_csv(output_file, index=False)
                
                logger.info(f"\n{'='*60}")
                logger.info(f"✅ REAL DATA SAVED")
                logger.info(f"{'='*60}")
                logger.info(f"File: {output_file}")
                logger.info(f"Total candles: {len(df):,}")
                logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
                logger.info(f"Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
                
                # Also save as the standard filename for easy use
                standard_file = output_dir / "BTCUSDT_15m_2019_2024_full.csv"
                df.to_csv(standard_file, index=False)
                logger.info(f"Also saved as: {standard_file}")
            else:
                logger.error("Failed to download data!")
                
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.info("Falling back to synthetic data...")
            args.real = False
    
    if not args.real:
        # Generate synthetic data (original behavior)
        logger.info("\n" + "="*60)
        logger.info("📊 GENERATING SYNTHETIC DATA")
        logger.info("="*60)
        logger.info("💡 Tip: Use --real flag to download actual Binance data")
        
        periods = [
            ("2019-01-01", "2019-12-31", "train_2019"),
            ("2020-01-01", "2020-12-31", "train_2020"),
            ("2021-01-01", "2021-12-31", "train_2021"),
            ("2022-01-01", "2022-12-31", "train_2022"),
            ("2023-01-01", "2023-12-31", "test_2023"),
            ("2024-01-01", "2024-12-31", "test_2024"),
        ]
        
        all_data = []
        
        for start, end, label in periods:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {label}: {start} to {end}")
            logger.info(f"{'='*60}")
            
            df = generate_realistic_btc_data(start, end, timeframe='15m')
            all_data.append(df)
            
            # Save individual file
            output_file = output_dir / f"BTCUSDT_15m_{label}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"💾 Saved to {output_file}")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_file = output_dir / "BTCUSDT_15m_2019_2024_full.csv"
        combined_df.to_csv(combined_file, index=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ ALL DATA SAVED")
        logger.info(f"{'='*60}")
        logger.info(f"Total candles: {len(combined_df):,}")
        logger.info(f"Combined file: {combined_file}")
        logger.info(f"Date range: {combined_df['time'].min()} to {combined_df['time'].max()}")
        logger.info(f"Price range: ${combined_df['close'].min():,.0f} - ${combined_df['close'].max():,.0f}")


if __name__ == "__main__":
    main()
