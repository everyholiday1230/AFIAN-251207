"""
Ensemble Model Training and Backtesting
========================================

Train ENSEMBLE model (XGBoost + TabNet + CatBoost) on 2019-2022, 
Backtest on 2023-2024

Features:
- Multi-model ensemble with voting
- XGBoost + TabNet + CatBoost
- Superior accuracy through model diversity

Usage:
    python scripts/ensemble_train_backtest_2023_2024.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime
import json

import numpy as np
import pandas as pd

from src.data_processing.custom_indicators import CustomIndicators
from src.data_processing.triple_barrier import TripleBarrierLabeler
from src.models.layer3_signal.ensemble_generator import EnsembleSignalGenerator
from src.utils.logger import get_logger

logger = get_logger("ensemble_train_backtest")


def load_data():
    """Load data from CSV."""
    logger.info("\n" + "=" * 80)
    logger.info("📂 LOADING DATA")
    logger.info("=" * 80)
    
    csv_file = Path("data/raw/BTCUSDT_15m_2019_2024_full.csv")
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Data file not found: {csv_file}")
    
    logger.info(f"Loading from {csv_file}")
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Split train/test
    train_start = datetime(2019, 1, 1)
    train_end = datetime(2022, 12, 31)
    test_start = datetime(2023, 1, 1)
    test_end = datetime(2024, 12, 31)
    
    train_df = df[(df['time'] >= train_start) & (df['time'] <= train_end)].copy()
    test_df = df[(df['time'] >= test_start) & (df['time'] <= test_end)].copy()
    
    logger.info(f"")
    logger.info(f"✅ Data loaded")
    logger.info(f"   📚 Training:  {len(train_df):,} candles ({train_df['time'].min()} to {train_df['time'].max()})")
    logger.info(f"   🧪 Testing:   {len(test_df):,} candles ({test_df['time'].min()} to {test_df['time'].max()})")
    logger.info(f"   💰 Price range (train): ${train_df['close'].min():,.0f} - ${train_df['close'].max():,.0f}")
    logger.info(f"   💰 Price range (test):  ${test_df['close'].min():,.0f} - ${test_df['close'].max():,.0f}")
    
    return train_df, test_df


def process_data(df, indicator_calculator, labeler, name="data"):
    """Calculate indicators and labels."""
    logger.info(f"\n🔧 Processing {name}...")
    
    # Calculate indicators
    df = indicator_calculator.calculate_all_indicators(df)
    logger.info(f"   ✅ Indicators calculated ({len(df):,} samples)")
    
    # Create labels
    df = labeler.create_labels(df)
    
    # Get label stats
    stats = labeler.get_label_statistics(df)
    logger.info(f"   ✅ Labels created ({stats['total_samples']:,} samples)")
    logger.info(f"      LONG: {stats['label_counts']['LONG']:,} ({stats['label_percentages']['LONG']:.1f}%)")
    logger.info(f"      SHORT: {stats['label_counts']['SHORT']:,} ({stats['label_percentages']['SHORT']:.1f}%)")
    logger.info(f"      NEUTRAL: {stats['label_counts']['NEUTRAL']:,} ({stats['label_percentages']['NEUTRAL']:.1f}%)")
    
    return df


def train_model(train_df, signal_generator):
    """Train the ensemble model."""
    logger.info("\n" + "=" * 80)
    logger.info("📚 TRAINING ENSEMBLE MODEL ON 2019-2022 DATA")
    logger.info("📚 Models: XGBoost + TabNet + CatBoost")
    logger.info("=" * 80)
    logger.info(f"Training samples: {len(train_df):,}")
    
    metrics = signal_generator.train(
        train_df,
        label_col='tb_label',
        test_size=0.2,
        balance_method='class_weight'
    )
    
    logger.info("")
    logger.info("✅ ENSEMBLE TRAINING COMPLETE")
    logger.info(f"   📊 Ensemble Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"   📊 Ensemble F1 Score:  {metrics['f1_score']:.4f}")
    
    # Log individual model metrics if available
    if 'individual_metrics' in metrics:
        logger.info("\n   📈 Individual Model Performance:")
        for model_name, model_metrics in metrics['individual_metrics'].items():
            if model_metrics:
                logger.info(f"      {model_name.upper()}: Acc={model_metrics['accuracy']:.4f}, F1={model_metrics['f1_score']:.4f}")
    
    return metrics


def backtest_continuous(test_df, signal_generator, indicator_calculator, initial_capital=10000.0, 
                       profit_target=0.015, stop_loss=0.005):
    """Run continuous backtest on test data."""
    logger.info("\n" + "=" * 80)
    logger.info("🧪 BACKTESTING ON 2023-2024 DATA")
    logger.info("=" * 80)
    
    # Get features
    feature_cols = indicator_calculator.get_feature_names()
    available_features = [col for col in feature_cols if col in test_df.columns]
    
    # Generate signals for all test data
    logger.info("Generating signals...")
    signals, confidence, _ = signal_generator.predict(test_df[available_features])
    
    # Run backtest
    logger.info("Running backtest...")
    capital = initial_capital
    position = None
    
    equity_curve = [capital]
    trades = []
    
    for i in range(len(test_df)):
        current_price = test_df.iloc[i]['close']
        current_time = test_df.iloc[i]['time']
        signal = signals[i]
        conf = confidence[i]
        
        # Check open position
        if position is not None:
            # Calculate P&L
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            # Exit conditions
            should_exit = (
                pnl_pct >= profit_target or
                pnl_pct <= -stop_loss or
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
        
        # Enter new position
        if position is None and signal in ['LONG', 'SHORT'] and conf >= 0.65:
            # Simple position sizing: 8% of capital
            position_size = capital * 0.08
            
            position = {
                'side': signal,
                'entry_price': current_price,
                'entry_time': current_time,
                'size': position_size
            }
        
        equity_curve.append(capital)
    
    # Close any remaining position
    if position is not None:
        current_price = test_df.iloc[-1]['close']
        current_time = test_df.iloc[-1]['time']
        
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
    total_return = (capital - initial_capital) / initial_capital
    
    if len(trades) > 0:
        returns = [t['pnl'] / initial_capital for t in trades]
        win_trades = [t for t in trades if t['pnl'] > 0]
        lose_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(win_trades) / len(trades)
        
        # Sharpe ratio
        if np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        peak = initial_capital
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Average win/loss
        avg_win = np.mean([t['pnl'] for t in win_trades]) if len(win_trades) > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in lose_trades]) if len(lose_trades) > 0 else 0
        
        # Profit factor
        total_profit = sum([t['pnl'] for t in win_trades])
        total_loss = abs(sum([t['pnl'] for t in lose_trades]))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
    else:
        win_rate = 0
        sharpe_ratio = 0
        max_dd = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        win_trades = []
        lose_trades = []
    
    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 BACKTEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"   💰 Initial Capital:      ${initial_capital:,.2f}")
    logger.info(f"   💰 Final Capital:        ${capital:,.2f}")
    logger.info(f"   📈 Total Return:         {total_return:.2%}")
    logger.info(f"   📊 Sharpe Ratio:         {sharpe_ratio:.3f}")
    logger.info(f"   📉 Max Drawdown:         {max_dd:.2%}")
    logger.info(f"   ✅ Win Rate:             {win_rate:.2%}")
    logger.info(f"   🔢 Total Trades:         {len(trades)}")
    logger.info(f"   ✅ Winning Trades:       {len(win_trades)}")
    logger.info(f"   ❌ Losing Trades:        {len(lose_trades)}")
    logger.info(f"   💵 Average Win:          ${avg_win:,.2f}")
    logger.info(f"   💸 Average Loss:         ${avg_loss:,.2f}")
    logger.info(f"   📊 Profit Factor:        {profit_factor:.2f}")
    logger.info("=" * 80)
    
    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'winning_trades': len(win_trades),
        'losing_trades': len(lose_trades),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'equity_curve': equity_curve,
        'trades': trades
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train on 2019-2022, backtest on 2023-2024')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--profit', type=float, default=0.015, help='Profit target (default: 1.5%)')
    parser.add_argument('--stoploss', type=float, default=0.005, help='Stop loss (default: 0.5%)')
    
    args = parser.parse_args()
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 ENSEMBLE TRAINING: 2019-2022, BACKTEST: 2023-2024")
    logger.info("🤖 Models: XGBoost + TabNet + CatBoost")
    logger.info("=" * 80)
    logger.info(f"Initial Capital: ${args.capital:,.2f}")
    logger.info(f"Profit Target:   {args.profit:.2%}")
    logger.info(f"Stop Loss:       {args.stoploss:.2%}")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Load data
        train_df, test_df = load_data()
        
        # Initialize components
        indicator_calculator = CustomIndicators()
        labeler = TripleBarrierLabeler(
            profit_target=args.profit,
            stop_loss=args.stoploss,
            time_limit_minutes=60
        )
        signal_generator = EnsembleSignalGenerator()  # Use ensemble model!
        
        # Process training data
        logger.info("\n" + "=" * 80)
        logger.info("📚 PROCESSING TRAINING DATA")
        logger.info("=" * 80)
        train_df = process_data(train_df, indicator_calculator, labeler, "training data")
        
        # Process testing data
        logger.info("\n" + "=" * 80)
        logger.info("🧪 PROCESSING TESTING DATA")
        logger.info("=" * 80)
        test_df = process_data(test_df, indicator_calculator, labeler, "testing data")
        
        # Train model
        train_metrics = train_model(train_df, signal_generator)
        
        # Backtest
        backtest_results = backtest_continuous(
            test_df, 
            signal_generator, 
            indicator_calculator,
            initial_capital=args.capital,
            profit_target=args.profit,
            stop_loss=args.stoploss
        )
        
        # Save results
        output_dir = Path('backtest_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f'ensemble_train_2019-2022_test_2023-2024_{timestamp}.json'
        
        # Filter out dataframe and model from train_metrics before saving
        train_metrics_filtered = {
            k: v for k, v in train_metrics.items() 
            if not isinstance(v, (pd.DataFrame, object)) or isinstance(v, (int, float, str, list, dict))
        }
        # Remove non-serializable keys
        if 'model' in train_metrics_filtered:
            del train_metrics_filtered['model']
        
        results = {
            'train_period': {
                'start': '2019-01-01',
                'end': '2022-12-31',
                'samples': len(train_df),
                'metrics': train_metrics_filtered
            },
            'test_period': {
                'start': '2023-01-01',
                'end': '2024-12-31',
                'samples': len(test_df)
            },
            'backtest_results': backtest_results,
            'parameters': {
                'initial_capital': args.capital,
                'profit_target': args.profit,
                'stop_loss': args.stoploss
            }
        }
        
        # Make serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_results = make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📄 Results saved to: {results_file}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"⏱️  Duration: {duration:.1f}s ({duration/60:.1f}m)")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
