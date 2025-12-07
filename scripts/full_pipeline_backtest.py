"""
Complete Trading Pipeline with All Layers
==========================================

Layer 1: Data Processing (Indicators)
Layer 2: Market Regime Detection  
Layer 3: Signal Generation (Ensemble AI)
Layer 4: Dynamic Position Sizing

Usage:
    python scripts/full_pipeline_backtest.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime
import json
import numpy as np
import pandas as pd

from src.data_processing.custom_indicators import CustomIndicators
from src.data_processing.triple_barrier import TripleBarrierLabeler
from src.models.layer2_regime.regime_detector import RegimeDetector
from src.models.layer3_signal.ensemble_generator import EnsembleSignalGenerator
from src.models.layer4_position.position_sizer import PositionSizer
from src.utils.logger import get_logger

logger = get_logger("full_pipeline")


def load_data():
    """Load training and testing data."""
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
    
    return train_df, test_df


def process_data_with_all_layers(df, indicator_calculator, labeler, regime_detector, name="data"):
    """Process data through all layers."""
    logger.info(f"\n🔧 Processing {name} through ALL LAYERS...")
    
    # Layer 1: Calculate indicators
    df = indicator_calculator.calculate_all_indicators(df)
    logger.info(f"   ✅ Layer 1: Indicators calculated")
    
    # Layer 2: Detect regime
    df = regime_detector.detect_regime(df)
    logger.info(f"   ✅ Layer 2: Regime detected")
    
    # Create labels (for training)
    df = labeler.create_labels(df)
    logger.info(f"   ✅ Triple Barrier labels created")
    
    return df


def run_full_pipeline_backtest(
    test_df,
    signal_generator,
    indicator_calculator,
    regime_detector,
    position_sizer,
    initial_capital=10000.0,
    profit_target=0.015,
    stop_loss=0.005
):
    """Run complete backtest with all layers."""
    logger.info("\n" + "=" * 80)
    logger.info("🚀 RUNNING FULL PIPELINE BACKTEST")
    logger.info("=" * 80)
    logger.info("  Layer 1: Custom Indicators ✓")
    logger.info("  Layer 2: Regime Detection ✓")
    logger.info("  Layer 3: Ensemble AI (XGBoost+TabNet+CatBoost) ✓")
    logger.info("  Layer 4: Dynamic Position Sizing ✓")
    logger.info("=" * 80)
    
    # Get features
    feature_cols = indicator_calculator.get_feature_names()
    available_features = [col for col in feature_cols if col in test_df.columns]
    
    # Generate signals (Layer 3)
    logger.info("\n📊 Layer 3: Generating trading signals...")
    signals, confidence, _ = signal_generator.predict(test_df[available_features])
    
    # Run backtest
    logger.info("💰 Starting backtest with dynamic position sizing...")
    capital = initial_capital
    position = None
    
    equity_curve = [capital]
    trades = []
    
    for i in range(len(test_df)):
        current_price = test_df.iloc[i]['close']
        current_time = test_df.iloc[i]['time']
        signal = signals[i]
        conf = confidence[i]
        regime = test_df.iloc[i]['regime']
        volatility = test_df.iloc[i]['atr_pct'] if 'atr_pct' in test_df.columns else None
        
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
                    'size': position['size'],
                    'size_pct': position['size_pct'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'confidence': position['confidence'],
                    'regime': position['regime']
                })
                
                position = None
        
        # Enter new position (Layer 3 + Layer 4)
        if position is None and signal in ['LONG', 'SHORT'] and conf >= 0.65:
            # Layer 4: Calculate dynamic position size
            pos_info = position_sizer.calculate_position(
                account_balance=capital,
                signal_confidence=conf,
                current_price=current_price,
                stop_loss_pct=stop_loss,
                volatility=volatility,
                regime=regime
            )
            
            position_size = pos_info['position_value']
            
            position = {
                'side': signal,
                'entry_price': current_price,
                'entry_time': current_time,
                'size': position_size,
                'size_pct': pos_info['position_size_pct'],
                'confidence': conf,
                'regime': regime
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
            'size': position['size'],
            'size_pct': position['size_pct'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'confidence': position['confidence'],
            'regime': position['regime']
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
        
        # Average position size
        avg_position_size = np.mean([t['size_pct'] for t in trades])
        
    else:
        win_rate = 0
        sharpe_ratio = 0
        max_dd = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        win_trades = []
        lose_trades = []
        avg_position_size = 0
    
    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 FULL PIPELINE BACKTEST RESULTS")
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
    logger.info(f"   📏 Avg Position Size:    {avg_position_size:.2%}")
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
        'avg_position_size': avg_position_size,
        'equity_curve': equity_curve,
        'trades': trades
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Full Pipeline Backtest (All Layers)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--sizing-method', type=str, default='HYBRID', 
                       choices=['FIXED', 'KELLY', 'VOLATILITY', 'CONFIDENCE', 'HYBRID'],
                       help='Position sizing method')
    
    args = parser.parse_args()
    
    logger.info("\n" + "=" * 80)
    logger.info("🎯 COMPLETE TRADING SYSTEM - ALL LAYERS")
    logger.info("=" * 80)
    logger.info(f"Initial Capital: ${args.capital:,.2f}")
    logger.info(f"Position Sizing: {args.sizing_method}")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Load data
        train_df, test_df = load_data()
        
        # Initialize all layers
        indicator_calculator = CustomIndicators()
        labeler = TripleBarrierLabeler(
            profit_target=0.015,
            stop_loss=0.005,
            time_limit_minutes=60
        )
        regime_detector = RegimeDetector()
        signal_generator = EnsembleSignalGenerator()
        position_sizer = PositionSizer(method=args.sizing_method)
        
        # Process training data
        logger.info("\n" + "=" * 80)
        logger.info("📚 PROCESSING TRAINING DATA")
        logger.info("=" * 80)
        train_df = process_data_with_all_layers(
            train_df, indicator_calculator, labeler, regime_detector, "training data"
        )
        
        # Process testing data
        logger.info("\n" + "=" * 80)
        logger.info("🧪 PROCESSING TESTING DATA")
        logger.info("=" * 80)
        test_df = process_data_with_all_layers(
            test_df, indicator_calculator, labeler, regime_detector, "testing data"
        )
        
        # Train Layer 3 (Ensemble AI)
        logger.info("\n" + "=" * 80)
        logger.info("🤖 TRAINING LAYER 3: ENSEMBLE AI")
        logger.info("=" * 80)
        train_metrics = signal_generator.train(
            train_df,
            label_col='tb_label',
            test_size=0.2,
            balance_method='class_weight'
        )
        
        logger.info(f"\n✅ Training completed")
        logger.info(f"   Ensemble Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"   Ensemble F1 Score: {train_metrics['f1_score']:.4f}")
        
        # Run full pipeline backtest
        backtest_results = run_full_pipeline_backtest(
            test_df,
            signal_generator,
            indicator_calculator,
            regime_detector,
            position_sizer,
            initial_capital=args.capital
        )
        
        # Save results
        output_dir = Path('backtest_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f'full_pipeline_{timestamp}.json'
        
        # Prepare results for saving
        results = {
            'pipeline': 'Complete (Layer 1+2+3+4)',
            'layers': {
                'layer1': 'Custom Indicators',
                'layer2': 'Market Regime Detection',
                'layer3': 'Ensemble AI (XGBoost+TabNet+CatBoost)',
                'layer4': f'Dynamic Position Sizing ({args.sizing_method})'
            },
            'train_period': {
                'start': '2019-01-01',
                'end': '2022-12-31',
                'samples': len(train_df)
            },
            'test_period': {
                'start': '2023-01-01',
                'end': '2024-12-31',
                'samples': len(test_df)
            },
            'backtest_results': {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in backtest_results.items()
                if k not in ['equity_curve', 'trades']
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
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
