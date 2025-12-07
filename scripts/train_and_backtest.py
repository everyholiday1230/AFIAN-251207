"""
Complete Training and Backtesting Pipeline
===========================================

This script:
1. Collects historical data from Binance
2. Calculates custom indicators (12 AI Oscillator + 3 Impulse MACD)
3. Creates Triple Barrier labels
4. Trains XGBoost signal generator
5. Performs Walk-Forward backtesting
6. Generates comprehensive performance report

Usage:
    python scripts/train_and_backtest.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_collection.binance_client import BinanceClient
from src.data_processing.custom_indicators import CustomIndicators
from src.data_processing.triple_barrier import TripleBarrierLabeler
from src.models.layer3_signal.signal_generator import SignalGenerator
from src.risk_management.risk_manager import RiskManager
from src.utils.logger import get_logger

logger = get_logger("train_backtest")


class CompletePipeline:
    """Complete training and backtesting pipeline."""
    
    def __init__(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '15m',
        start_date: str = '2023-01-01',
        end_date: str = '2024-12-31',
        initial_capital: float = 10000.0,
        # Triple Barrier parameters
        profit_target: float = 0.015,
        stop_loss: float = 0.005,
        time_limit_minutes: int = 60,
        # Walk-Forward parameters
        train_window_days: int = 180,
        test_window_days: int = 30,
        step_days: int = 30,
    ):
        """Initialize pipeline with parameters."""
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.initial_capital = initial_capital
        
        # Triple Barrier
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.time_limit_minutes = time_limit_minutes
        
        # Walk-Forward
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        
        # Components
        self.binance_client = None
        self.indicator_calculator = CustomIndicators()
        self.labeler = TripleBarrierLabeler(
            profit_target=profit_target,
            stop_loss=stop_loss,
            time_limit_minutes=time_limit_minutes
        )
        self.signal_generator = SignalGenerator(model_type='xgboost')
        self.risk_manager = RiskManager()
        
        logger.info("=" * 60)
        logger.info("Pipeline initialized")
        logger.info("=" * 60)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info(f"Triple Barrier: +{profit_target:.1%} / -{stop_loss:.1%} / {time_limit_minutes}min")
        logger.info(f"Walk-Forward: {train_window_days}d train / {test_window_days}d test / {step_days}d step")
        logger.info("=" * 60)
    
    def step1_collect_data(self) -> pd.DataFrame:
        """Step 1: Collect historical data from CSV files or Binance."""
        logger.info("STEP 1: Collecting historical data...")
        
        # Try to load from CSV file first
        csv_file = Path("data/raw/BTCUSDT_15m_2019_2024_full.csv")
        if csv_file.exists():
            logger.info(f"📂 Loading data from {csv_file}")
            df = pd.read_csv(csv_file)
            df['time'] = pd.to_datetime(df['time'])
            
            # Filter by date range
            df = df[(df['time'] >= self.start_date) & (df['time'] <= self.end_date)]
            
            logger.info(f"✅ Loaded {len(df)} candles from CSV")
            logger.info(f"   Date range: {df['time'].min()} to {df['time'].max()}")
            
            return df
        
        try:
            # Try to fetch real data from Binance
            logger.info("Attempting to fetch real data from Binance...")
            self.binance_client = BinanceClient(testnet=False)
            
            df = self.binance_client.fetch_historical_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            logger.info(f"✅ Collected {len(df)} real candles from Binance")
            logger.info(f"   Date range: {df['time'].min()} to {df['time'].max()}")
            
            return df
            
        except Exception as e:
            # If Binance fails, generate realistic sample data
            logger.warning(f"⚠️ Failed to fetch from Binance: {e}")
            logger.info("📊 Generating realistic sample data instead...")
            
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate realistic BTC/USDT sample data for backtesting."""
        
        # Generate timestamps
        if self.timeframe == '15m':
            freq = '15T'
        elif self.timeframe == '1h':
            freq = '1H'
        elif self.timeframe == '5m':
            freq = '5T'
        else:
            freq = '15T'
        
        timestamps = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=freq
        )
        
        n = len(timestamps)
        logger.info(f"Generating {n} sample candles...")
        
        # Realistic BTC price simulation (2024: $40k-$70k range)
        np.random.seed(42)  # Reproducible
        
        # Generate realistic price movement
        base_price = 50000  # Starting price
        trend = np.linspace(0, 20000, n)  # Upward trend
        volatility = np.random.randn(n).cumsum() * 1000  # Random walk
        seasonality = 5000 * np.sin(np.linspace(0, 8*np.pi, n))  # Cycles
        
        close = base_price + trend + volatility + seasonality
        close = np.maximum(close, 30000)  # Floor price
        close = np.minimum(close, 80000)  # Ceiling price
        
        # Generate OHLCV
        high = close * (1 + np.abs(np.random.randn(n)) * 0.002)
        low = close * (1 - np.abs(np.random.randn(n)) * 0.002)
        open_price = np.roll(close, 1)
        open_price[0] = close[0]
        volume = np.random.uniform(100, 1000, n)
        
        df = pd.DataFrame({
            'time': timestamps,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        logger.info(f"✅ Generated {len(df)} sample candles")
        logger.info(f"   Date range: {df['time'].min()} to {df['time'].max()}")
        logger.info(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
        
        return df
    
    def step2_calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Calculate custom indicators."""
        logger.info("STEP 2: Calculating custom indicators...")
        
        df_indicators = self.indicator_calculator.calculate_all_indicators(df)
        
        feature_names = self.indicator_calculator.get_feature_names()
        logger.info(f"✅ Calculated {len(feature_names)} indicators")
        
        return df_indicators
    
    def step3_create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 3: Create Triple Barrier labels."""
        logger.info("STEP 3: Creating Triple Barrier labels...")
        
        df_labeled = self.labeler.create_labels(df)
        
        # Get statistics
        stats = self.labeler.get_label_statistics(df_labeled)
        
        logger.info(f"✅ Labels created")
        logger.info(f"   Total samples: {stats['total_samples']}")
        logger.info(f"   Label distribution:")
        for label, count in stats['label_counts'].items():
            pct = stats['label_percentages'][label]
            logger.info(f"     {label}: {count} ({pct:.1f}%)")
        
        return df_labeled
    
    def step4_walk_forward_backtest(self, df: pd.DataFrame) -> dict:
        """Step 4: Perform Walk-Forward backtesting."""
        logger.info("STEP 4: Starting Walk-Forward backtesting...")
        logger.info("=" * 60)
        
        # Get feature columns
        feature_cols = self.indicator_calculator.get_feature_names()
        
        # Walk-Forward splits
        current_date = self.start_date + timedelta(days=self.train_window_days)
        end_date = self.end_date
        
        fold_num = 0
        all_results = []
        
        while current_date + timedelta(days=self.test_window_days) <= end_date:
            fold_num += 1
            
            # Define train and test periods
            train_start = current_date - timedelta(days=self.train_window_days)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=self.test_window_days)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold_num}")
            logger.info(f"{'='*60}")
            logger.info(f"Train: {train_start.date()} to {train_end.date()}")
            logger.info(f"Test:  {test_start.date()} to {test_end.date()}")
            
            # Split data
            train_df = df[(df['time'] >= train_start) & (df['time'] < train_end)].copy()
            test_df = df[(df['time'] >= test_start) & (df['time'] < test_end)].copy()
            
            logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
            
            if len(train_df) < 1000 or len(test_df) < 100:
                logger.warning(f"Insufficient data for fold {fold_num}, skipping...")
                current_date += timedelta(days=self.step_days)
                continue
            
            # Train model
            logger.info("Training model...")
            metrics = self.signal_generator.train(
                train_df,
                label_col='tb_label',
                test_size=0.2,
                balance_method='class_weight'
            )
            
            logger.info(f"Training accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Training F1: {metrics['f1_score']:.4f}")
            
            # Backtest on test period
            logger.info("Backtesting on test period...")
            fold_results = self._backtest_period(test_df, feature_cols)
            
            fold_results['fold'] = fold_num
            fold_results['train_start'] = train_start
            fold_results['train_end'] = train_end
            fold_results['test_start'] = test_start
            fold_results['test_end'] = test_end
            fold_results['train_accuracy'] = metrics['accuracy']
            fold_results['train_f1'] = metrics['f1_score']
            
            all_results.append(fold_results)
            
            logger.info(f"\nFold {fold_num} Results:")
            logger.info(f"  Total Return: {fold_results['total_return']:.2%}")
            logger.info(f"  Sharpe Ratio: {fold_results['sharpe_ratio']:.3f}")
            logger.info(f"  Max Drawdown: {fold_results['max_drawdown']:.2%}")
            logger.info(f"  Win Rate: {fold_results['win_rate']:.2%}")
            logger.info(f"  Total Trades: {fold_results['total_trades']}")
            
            # Move to next fold
            current_date += timedelta(days=self.step_days)
        
        # Aggregate results
        logger.info("\n" + "=" * 60)
        logger.info("WALK-FORWARD BACKTEST COMPLETE")
        logger.info("=" * 60)
        
        summary = self._aggregate_results(all_results)
        
        return {
            'fold_results': all_results,
            'summary': summary
        }
    
    def _backtest_period(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """Backtest a single period."""
        # Get available features (exclude any non-feature columns like 'impulse_color')
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Generate signals
        signals, confidence, _ = self.signal_generator.predict(df[available_features])
        
        # Initialize portfolio
        capital = self.initial_capital
        position = None  # {'side': 'LONG'/'SHORT', 'entry_price': float, 'size': float}
        
        equity_curve = [capital]
        trades = []
        
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            signal = signals[i]
            conf = confidence[i]
            
            # Check if we have an open position
            if position is not None:
                # Check exit conditions (simplified)
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:  # SHORT
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                # Exit conditions: profit target, stop loss, or opposite signal
                should_exit = (
                    pnl_pct >= self.profit_target or
                    pnl_pct <= -self.stop_loss or
                    (position['side'] == 'LONG' and signal == 'SHORT') or
                    (position['side'] == 'SHORT' and signal == 'LONG')
                )
                
                if should_exit:
                    # Close position
                    pnl = position['size'] * pnl_pct
                    capital += pnl
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': df.iloc[i]['time'],
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'return': pnl / capital  # Return on capital
                    })
                    
                    position = None
            
            # Enter new position if signal and no position
            if position is None and signal in ['LONG', 'SHORT']:
                # Risk check
                can_trade, decision = self.risk_manager.evaluate_trade(
                    current_equity=capital,
                    signal_confidence=conf,
                    current_volatility=0.02,  # Simplified
                    avg_volatility=0.02,
                    open_positions=0
                )
                
                if can_trade:
                    # Enter position
                    position_size = capital * decision.recommended_position_size
                    
                    position = {
                        'side': signal,
                        'entry_price': current_price,
                        'entry_time': df.iloc[i]['time'],
                        'size': position_size
                    }
            
            equity_curve.append(capital)
        
        # Close any remaining position
        if position is not None:
            current_price = df.iloc[-1]['close']
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            pnl = position['size'] * pnl_pct
            capital += pnl
            
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df.iloc[-1]['time'],
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'return': pnl / capital
            })
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        if len(trades) > 0:
            returns = [t['return'] for t in trades]
            win_trades = [t for t in trades if t['pnl'] > 0]
            
            win_rate = len(win_trades) / len(trades)
            
            # Sharpe ratio (annualized)
            if np.std(returns) > 0:
                sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            peak = self.initial_capital
            max_dd = 0
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            win_rate = 0
            sharpe_ratio = 0
            max_dd = 0
        
        return {
            'total_return': total_return,
            'final_capital': capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def _aggregate_results(self, all_results: list) -> dict:
        """Aggregate results from all folds."""
        if not all_results:
            return {}
        
        # Calculate overall metrics
        total_returns = [r['total_return'] for r in all_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in all_results]
        max_drawdowns = [r['max_drawdown'] for r in all_results]
        win_rates = [r['win_rate'] for r in all_results]
        
        summary = {
            'total_folds': len(all_results),
            'avg_return_per_fold': np.mean(total_returns),
            'std_return_per_fold': np.std(total_returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'best_fold_return': max(total_returns),
            'worst_fold_return': min(total_returns),
            'positive_folds': sum(1 for r in total_returns if r > 0),
            'negative_folds': sum(1 for r in total_returns if r < 0),
        }
        
        logger.info("\nOVERALL SUMMARY:")
        logger.info(f"  Total Folds: {summary['total_folds']}")
        logger.info(f"  Avg Return/Fold: {summary['avg_return_per_fold']:.2%} ± {summary['std_return_per_fold']:.2%}")
        logger.info(f"  Avg Sharpe: {summary['avg_sharpe']:.3f}")
        logger.info(f"  Avg Max DD: {summary['avg_max_drawdown']:.2%}")
        logger.info(f"  Avg Win Rate: {summary['avg_win_rate']:.2%}")
        logger.info(f"  Positive/Negative Folds: {summary['positive_folds']}/{summary['negative_folds']}")
        
        return summary
    
    def run_complete_pipeline(self):
        """Run the complete pipeline."""
        logger.info("\n" + "=" * 60)
        logger.info("STARTING COMPLETE PIPELINE")
        logger.info("=" * 60 + "\n")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Collect data
            df = self.step1_collect_data()
            
            # Step 2: Calculate indicators
            df = self.step2_calculate_indicators(df)
            
            # Step 3: Create labels
            df = self.step3_create_labels(df)
            
            # Step 4: Walk-Forward backtest
            results = self.step4_walk_forward_backtest(df)
            
            # Save results
            output_dir = Path('backtest_results')
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = output_dir / f'backtest_{timestamp}.json'
            
            # Convert datetime objects to strings for JSON
            serializable_results = self._make_serializable(results)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"\n✅ Results saved to {results_file}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info("=" * 60)
            
            return results
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train and backtest trading system')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='15m', help='Timeframe')
    parser.add_argument('--start', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = CompletePipeline(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )
    
    results = pipeline.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    main()
