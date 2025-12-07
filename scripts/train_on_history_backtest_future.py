"""
Train on Historical Data and Backtest on Future Data
====================================================

This script:
1. Trains the model on historical data (2019-2022)
2. Backtests the trained model on future data (2023-2024)
3. Uses Walk-Forward approach within the test period

Usage:
    python scripts/train_on_history_backtest_future.py
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

from src.data_collection.binance_client import BinanceClient
from src.data_processing.custom_indicators import CustomIndicators
from src.data_processing.triple_barrier import TripleBarrierLabeler
from src.models.layer3_signal.signal_generator import SignalGenerator
from src.risk_management.risk_manager import RiskManager
from src.utils.logger import get_logger

logger = get_logger("train_backtest_split")


class TrainBacktestPipeline:
    """Train on historical data, backtest on future data."""
    
    def __init__(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '15m',
        train_start: str = '2019-01-01',
        train_end: str = '2022-12-31',
        test_start: str = '2023-01-01',
        test_end: str = '2024-12-31',
        initial_capital: float = 10000.0,
        # Triple Barrier parameters
        profit_target: float = 0.015,
        stop_loss: float = 0.005,
        time_limit_minutes: int = 60,
        # Walk-Forward parameters for test period
        test_window_days: int = 30,
        step_days: int = 30,
    ):
        """Initialize pipeline with parameters."""
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_start = datetime.strptime(train_start, '%Y-%m-%d')
        self.train_end = datetime.strptime(train_end, '%Y-%m-%d')
        self.test_start = datetime.strptime(test_start, '%Y-%m-%d')
        self.test_end = datetime.strptime(test_end, '%Y-%m-%d')
        self.initial_capital = initial_capital
        
        # Triple Barrier
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.time_limit_minutes = time_limit_minutes
        
        # Walk-Forward for test period
        self.test_window_days = test_window_days
        self.step_days = step_days
        
        # Components
        self.indicator_calculator = CustomIndicators()
        self.labeler = TripleBarrierLabeler(
            profit_target=profit_target,
            stop_loss=stop_loss,
            time_limit_minutes=time_limit_minutes
        )
        self.signal_generator = SignalGenerator(model_type='xgboost')
        self.risk_manager = RiskManager()
        
        logger.info("=" * 80)
        logger.info("🚀 TRAIN ON HISTORICAL DATA, BACKTEST ON FUTURE DATA")
        logger.info("=" * 80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"")
        logger.info(f"📚 TRAINING PERIOD: {train_start} to {train_end}")
        logger.info(f"🧪 TESTING PERIOD:  {test_start} to {test_end}")
        logger.info(f"")
        logger.info(f"💰 Initial capital: ${initial_capital:,.2f}")
        logger.info(f"🎯 Triple Barrier: Profit={profit_target:.2%}, StopLoss={stop_loss:.2%}, TimeLimit={time_limit_minutes}min")
        logger.info(f"🔄 Walk-Forward in test: {test_window_days}d window / {step_days}d step")
        logger.info("=" * 80)
    
    def load_data(self) -> tuple:
        """Load data from CSV files."""
        logger.info("\n📂 STEP 1: Loading data from CSV files...")
        
        csv_file = Path("data/raw/BTCUSDT_15m_2019_2024_full.csv")
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Data file not found: {csv_file}")
        
        logger.info(f"   Loading from {csv_file}")
        df = pd.read_csv(csv_file)
        df['time'] = pd.to_datetime(df['time'])
        
        # Split into train and test
        train_df = df[(df['time'] >= self.train_start) & (df['time'] <= self.train_end)].copy()
        test_df = df[(df['time'] >= self.test_start) & (df['time'] <= self.test_end)].copy()
        
        logger.info(f"")
        logger.info(f"✅ Data loaded successfully")
        logger.info(f"   📚 Training data: {len(train_df):,} candles ({train_df['time'].min()} to {train_df['time'].max()})")
        logger.info(f"   🧪 Testing data:  {len(test_df):,} candles ({test_df['time'].min()} to {test_df['time'].max()})")
        logger.info(f"   📊 Price range (train): ${train_df['close'].min():,.0f} - ${train_df['close'].max():,.0f}")
        logger.info(f"   📊 Price range (test):  ${test_df['close'].min():,.0f} - ${test_df['close'].max():,.0f}")
        
        return train_df, test_df
    
    def calculate_indicators(self, df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
        """Calculate custom indicators."""
        logger.info(f"\n🔧 Calculating indicators for {name}...")
        
        df_indicators = self.indicator_calculator.calculate_all_indicators(df)
        
        feature_names = self.indicator_calculator.get_feature_names()
        logger.info(f"   ✅ Calculated {len(feature_names)} indicators")
        logger.info(f"   📊 Valid samples after NaN removal: {len(df_indicators):,}")
        
        return df_indicators
    
    def create_labels(self, df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
        """Create Triple Barrier labels."""
        logger.info(f"\n🏷️  Creating labels for {name}...")
        
        df_labeled = self.labeler.create_labels(df)
        
        # Get statistics
        stats = self.labeler.get_label_statistics(df_labeled)
        
        logger.info(f"   ✅ Labels created")
        logger.info(f"   📊 Total samples: {stats['total_samples']:,}")
        logger.info(f"   📊 Label distribution:")
        for label, count in stats['label_counts'].items():
            pct = stats['label_percentages'][label]
            logger.info(f"      {label}: {count:,} ({pct:.1f}%)")
        
        return df_labeled
    
    def train_model(self, train_df: pd.DataFrame):
        """Train the model on historical data."""
        logger.info("\n" + "=" * 80)
        logger.info("📚 STEP 2: TRAINING MODEL ON HISTORICAL DATA (2019-2022)")
        logger.info("=" * 80)
        
        logger.info(f"Training samples: {len(train_df):,}")
        logger.info(f"Training period: {train_df['time'].min()} to {train_df['time'].max()}")
        
        # Train model
        logger.info("\n🔄 Training XGBoost model...")
        metrics = self.signal_generator.train(
            train_df,
            label_col='tb_label',
            test_size=0.2,
            balance_method='class_weight'
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ MODEL TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"   📊 Test Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   📊 F1 Score:       {metrics['f1_score']:.4f}")
        if 'confidence_filtered_accuracy' in metrics:
            logger.info(f"   📊 Filtered Accuracy: {metrics['confidence_filtered_accuracy']:.4f}")
        
        # Show top features
        if hasattr(self.signal_generator.model, 'feature_importances_'):
            feature_names = self.indicator_calculator.get_feature_names()
            importances = self.signal_generator.model.feature_importances_
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            
            logger.info(f"\n   🔝 Top 10 Important Features:")
            for i in range(min(10, len(feature_names))):
                idx = indices[i]
                logger.info(f"      {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        logger.info("=" * 80)
        
        return metrics
    
    def backtest_period(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> dict:
        """Backtest a single period."""
        period_df = df[(df['time'] >= start_date) & (df['time'] <= end_date)].copy()
        
        if len(period_df) == 0:
            return None
        
        # Get feature columns
        feature_cols = self.indicator_calculator.get_feature_names()
        available_features = [col for col in feature_cols if col in period_df.columns]
        
        # Generate signals
        signals, confidence, _ = self.signal_generator.predict(period_df[available_features])
        
        # Initialize portfolio
        capital = self.initial_capital
        position = None  # {'side': 'LONG'/'SHORT', 'entry_price': float, 'size': float}
        
        equity_curve = [capital]
        trades = []
        
        for i in range(len(period_df)):
            current_price = period_df.iloc[i]['close']
            current_time = period_df.iloc[i]['time']
            signal = signals[i]
            conf = confidence[i]
            
            # Check if we have an open position
            if position is not None:
                # Calculate P&L
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:  # SHORT
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                # Exit conditions
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
                        'exit_time': current_time,
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                    })
                    
                    position = None
            
            # Enter new position if signal and no position
            if position is None and signal in ['LONG', 'SHORT']:
                # Risk check
                can_trade, decision = self.risk_manager.evaluate_trade(
                    current_equity=capital,
                    signal_confidence=conf,
                    current_volatility=0.02,
                    avg_volatility=0.02,
                    open_positions=0
                )
                
                if can_trade:
                    # Enter position
                    position_size = capital * decision.recommended_position_size
                    
                    position = {
                        'side': signal,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'size': position_size
                    }
            
            equity_curve.append(capital)
        
        # Close any remaining position
        if position is not None:
            current_price = period_df.iloc[-1]['close']
            current_time = period_df.iloc[-1]['time']
            
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            pnl = position['size'] * pnl_pct
            capital += pnl
            
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
            })
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        if len(trades) > 0:
            returns = [t['pnl'] / self.initial_capital for t in trades]
            win_trades = [t for t in trades if t['pnl'] > 0]
            
            win_rate = len(win_trades) / len(trades)
            
            # Sharpe ratio
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
            'start_date': start_date,
            'end_date': end_date,
            'total_return': total_return,
            'final_capital': capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def walk_forward_backtest(self, test_df: pd.DataFrame) -> dict:
        """Perform walk-forward backtesting on test period."""
        logger.info("\n" + "=" * 80)
        logger.info("🧪 STEP 3: BACKTESTING ON FUTURE DATA (2023-2024)")
        logger.info("=" * 80)
        
        current_date = self.test_start
        end_date = self.test_end
        
        fold_num = 0
        all_results = []
        
        while current_date < end_date:
            fold_num += 1
            
            # Define test period
            period_start = current_date
            period_end = min(current_date + timedelta(days=self.test_window_days), end_date)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📅 Fold {fold_num}: {period_start.date()} to {period_end.date()}")
            logger.info(f"{'='*80}")
            
            # Backtest this period
            fold_results = self.backtest_period(test_df, period_start, period_end)
            
            if fold_results is None:
                logger.warning(f"⚠️  No data for this period, skipping...")
                current_date += timedelta(days=self.step_days)
                continue
            
            fold_results['fold'] = fold_num
            all_results.append(fold_results)
            
            logger.info(f"   💰 Total Return:   {fold_results['total_return']:.2%}")
            logger.info(f"   📊 Sharpe Ratio:   {fold_results['sharpe_ratio']:.3f}")
            logger.info(f"   📉 Max Drawdown:   {fold_results['max_drawdown']:.2%}")
            logger.info(f"   ✅ Win Rate:       {fold_results['win_rate']:.2%}")
            logger.info(f"   🔢 Total Trades:   {fold_results['total_trades']}")
            logger.info(f"   💵 Final Capital:  ${fold_results['final_capital']:,.2f}")
            
            # Move to next period
            current_date += timedelta(days=self.step_days)
        
        # Aggregate results
        logger.info("\n" + "=" * 80)
        logger.info("📊 BACKTEST COMPLETE - OVERALL SUMMARY")
        logger.info("=" * 80)
        
        summary = self._aggregate_results(all_results)
        
        return {
            'fold_results': all_results,
            'summary': summary
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
        total_trades_list = [r['total_trades'] for r in all_results]
        
        # Calculate cumulative return safely
        if len(total_returns) > 0:
            cumulative_return = np.prod([1 + r for r in total_returns]) - 1
        else:
            cumulative_return = 0.0
        
        summary = {
            'total_folds': len(all_results),
            'avg_return_per_fold': np.mean(total_returns) if len(total_returns) > 0 else 0.0,
            'std_return_per_fold': np.std(total_returns) if len(total_returns) > 0 else 0.0,
            'cumulative_return': cumulative_return,
            'avg_sharpe': np.mean(sharpe_ratios) if len(sharpe_ratios) > 0 else 0.0,
            'avg_max_drawdown': np.mean(max_drawdowns) if len(max_drawdowns) > 0 else 0.0,
            'avg_win_rate': np.mean(win_rates) if len(win_rates) > 0 else 0.0,
            'total_trades': sum(total_trades_list),
            'best_fold_return': max(total_returns) if len(total_returns) > 0 else 0.0,
            'worst_fold_return': min(total_returns) if len(total_returns) > 0 else 0.0,
            'positive_folds': sum(1 for r in total_returns if r > 0),
            'negative_folds': sum(1 for r in total_returns if r < 0),
        }
        
        logger.info(f"\n   📈 Total Folds:           {summary['total_folds']}")
        logger.info(f"   💰 Avg Return/Fold:       {summary['avg_return_per_fold']:.2%} ± {summary['std_return_per_fold']:.2%}")
        logger.info(f"   💵 Cumulative Return:     {summary['cumulative_return']:.2%}")
        logger.info(f"   📊 Avg Sharpe Ratio:      {summary['avg_sharpe']:.3f}")
        logger.info(f"   📉 Avg Max Drawdown:      {summary['avg_max_drawdown']:.2%}")
        logger.info(f"   ✅ Avg Win Rate:          {summary['avg_win_rate']:.2%}")
        logger.info(f"   🔢 Total Trades:          {summary['total_trades']}")
        logger.info(f"   🎯 Best Fold Return:      {summary['best_fold_return']:.2%}")
        logger.info(f"   ⚠️  Worst Fold Return:     {summary['worst_fold_return']:.2%}")
        logger.info(f"   ✅ Positive/Negative Folds: {summary['positive_folds']}/{summary['negative_folds']}")
        
        return summary
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING COMPLETE PIPELINE")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load data
            train_df, test_df = self.load_data()
            
            # Step 2: Calculate indicators for training data
            logger.info("\n" + "=" * 80)
            logger.info("📚 PROCESSING TRAINING DATA (2019-2022)")
            logger.info("=" * 80)
            train_df = self.calculate_indicators(train_df, "training data")
            train_df = self.create_labels(train_df, "training data")
            
            # Step 3: Calculate indicators for testing data
            logger.info("\n" + "=" * 80)
            logger.info("🧪 PROCESSING TESTING DATA (2023-2024)")
            logger.info("=" * 80)
            test_df = self.calculate_indicators(test_df, "testing data")
            test_df = self.create_labels(test_df, "testing data")
            
            # Step 4: Train model on historical data
            train_metrics = self.train_model(train_df)
            
            # Step 5: Backtest on future data
            backtest_results = self.walk_forward_backtest(test_df)
            
            # Save results
            output_dir = Path('backtest_results')
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = output_dir / f'backtest_2019-2022_train_{timestamp}.json'
            
            results = {
                'train_period': {
                    'start': self.train_start.isoformat(),
                    'end': self.train_end.isoformat(),
                    'samples': len(train_df),
                    'metrics': train_metrics
                },
                'test_period': {
                    'start': self.test_start.isoformat(),
                    'end': self.test_end.isoformat(),
                    'samples': len(test_df)
                },
                'backtest_results': backtest_results
            }
            
            # Convert to JSON-serializable format
            serializable_results = self._make_serializable(results)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ PIPELINE COMPLETE")
            logger.info("=" * 80)
            logger.info(f"📄 Results saved to: {results_file}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"⏱️  Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info("=" * 80)
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
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
    parser = argparse.ArgumentParser(description='Train on 2019-2022, backtest on 2023-2024')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='15m', help='Timeframe')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = TrainBacktestPipeline(
        symbol=args.symbol,
        timeframe=args.timeframe,
        train_start='2019-01-01',
        train_end='2022-12-31',
        test_start='2023-01-01',
        test_end='2024-12-31',
        initial_capital=args.capital,
        test_window_days=90,  # 3-month windows for testing
        step_days=90
    )
    
    results = pipeline.run_pipeline()
    
    return results


if __name__ == "__main__":
    main()
