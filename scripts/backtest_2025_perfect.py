#!/usr/bin/env python3
"""
🚀 2025년 완벽한 백테스트 스크립트
=================================

목적: 2019-2022 학습 데이터로 모델을 학습하고, 
     2025년 1월-11월 실제 Binance 데이터로 미래를 모르는 상태에서 백테스팅

특징:
- 미래 데이터 누수 방지 (No Look-Ahead Bias)
- 실제 거래 환경 시뮬레이션
- 순차적 신호 생성 및 거래 실행
- 완벽한 리스크 관리 (Stop Loss, Profit Target)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.custom_indicators import CustomIndicators
from src.data_processing.triple_barrier import TripleBarrierLabeler
from src.models.layer3_signal.ensemble_generator import EnsembleSignalGenerator


def load_training_data():
    """Load 2019-2022 training data"""
    logger.info("\n" + "="*80)
    logger.info("📚 LOADING TRAINING DATA (2019-2022)")
    logger.info("="*80)
    
    data_path = project_root / "data/raw/BTCUSDT_15m_2019_2024_full.csv"
    logger.info(f"Loading from {data_path}")
    
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # Split: 2019-2022 for training
    train_df = df[df['time'].dt.year <= 2022].copy().reset_index(drop=True)
    
    logger.info("")
    logger.info("✅ Training data loaded")
    logger.info(f"   📚 Training: {len(train_df):,} candles ({train_df['time'].min()} to {train_df['time'].max()})")
    logger.info(f"   💰 Price range: ${train_df['close'].min():,.0f} - ${train_df['close'].max():,.0f}")
    
    return train_df


def load_test_data_2025():
    """Load 2025 test data (Jan-Nov)"""
    logger.info("\n" + "="*80)
    logger.info("🧪 LOADING TEST DATA (2025 JAN-NOV)")
    logger.info("="*80)
    
    data_path = project_root / "data/raw/BTCUSDT_15m_2025_jan_nov.csv"
    logger.info(f"Loading from {data_path}")
    
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    logger.info("")
    logger.info("✅ Test data loaded")
    logger.info(f"   🧪 Testing: {len(df):,} candles ({df['time'].min()} to {df['time'].max()})")
    logger.info(f"   💰 Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    
    return df


def process_data(df, indicator_calculator, labeler, is_training=True):
    """Process data: calculate indicators and labels"""
    label = "training" if is_training else "testing"
    logger.info(f"\n🔧 Processing {label} data...")
    
    # Calculate indicators
    df = indicator_calculator.calculate_all_indicators(df)
    logger.info(f"   ✅ Indicators calculated ({len(df):,} samples)")
    
    # Create labels (only for training data metrics)
    if is_training:
        df = labeler.create_labels(df)
        
        # Label distribution
        label_counts = df['tb_label'].value_counts()
        total = len(df)
        logger.info(f"   ✅ Labels created ({total:,} samples)")
        logger.info(f"      LONG: {label_counts.get('LONG', 0):,} ({100*label_counts.get('LONG', 0)/total:.1f}%)")
        logger.info(f"      SHORT: {label_counts.get('SHORT', 0):,} ({100*label_counts.get('SHORT', 0)/total:.1f}%)")
        logger.info(f"      NEUTRAL: {label_counts.get('NEUTRAL', 0):,} ({100*label_counts.get('NEUTRAL', 0)/total:.1f}%)")
    
    return df


def train_ensemble_model(train_df, indicator_calculator, confidence_threshold=0.50):
    """Train ensemble model on 2019-2022 data"""
    logger.info("\n" + "="*80)
    logger.info("🚀 TRAINING ENSEMBLE MODEL (2019-2022)")
    logger.info("🤖 Models: XGBoost + TabNet + CatBoost")
    logger.info("="*80)
    logger.info(f"Training samples: {len(train_df):,}")
    logger.info(f"Confidence threshold: {confidence_threshold:.2%}")
    
    # Initialize ensemble
    signal_generator = EnsembleSignalGenerator(confidence_threshold=confidence_threshold)
    
    # Get features
    feature_cols = indicator_calculator.get_feature_names()
    available_features = [col for col in feature_cols if col in train_df.columns]
    
    # Train (include all TB columns for filtering)
    tb_columns = [col for col in train_df.columns if col.startswith('tb_')]
    train_cols = available_features + tb_columns
    metrics = signal_generator.train(
        train_df[train_cols], 
        'tb_label'
    )
    
    logger.info("")
    logger.info("✅ ENSEMBLE TRAINING COMPLETE")
    logger.info(f"   📊 Ensemble Accuracy:  {metrics.get('accuracy', 0):.4f}")
    logger.info(f"   📊 Ensemble F1 Score:  {metrics.get('f1_score', 0):.4f}")
    
    if 'individual_metrics' in metrics:
        logger.info("\n   📈 Individual Model Performance:")
        for model_name, model_metrics in metrics['individual_metrics'].items():
            logger.info(f"      {model_name.upper()}: Acc={model_metrics.get('accuracy', 0):.4f}, F1={model_metrics.get('f1_score', 0):.4f}")
    
    return signal_generator, available_features, metrics


def backtest_2025(test_df, signal_generator, available_features, initial_capital=10000.0, 
                  profit_target=0.015, stop_loss=0.005, confidence_threshold=0.50):
    """
    Run backtest on 2025 data WITHOUT look-ahead bias
    
    완벽한 백테스팅:
    - 순차적으로 캔들을 하나씩 처리
    - 각 시점에서 과거 데이터만 사용
    - 미래 데이터 절대 사용 안함
    - 실제 거래 환경 완벽 시뮬레이션
    """
    logger.info("\n" + "="*80)
    logger.info("🧪 BACKTESTING ON 2025 DATA (JAN-NOV)")
    logger.info("🔒 NO LOOK-AHEAD BIAS - 순차적 처리")
    logger.info("="*80)
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Profit Target:   {profit_target:.2%}")
    logger.info(f"Stop Loss:       {stop_loss:.2%}")
    logger.info(f"Confidence:      {confidence_threshold:.2%}")
    
    # Generate signals for entire test period
    logger.info("\n📊 Generating trading signals...")
    signals, confidence, proba = signal_generator.predict(test_df[available_features])
    logger.info(f"✅ Generated {len(signals):,} signals")
    
    # Signal distribution
    signal_counts = pd.Series(signals).value_counts()
    logger.info(f"\n📈 Signal Distribution:")
    for sig, count in signal_counts.items():
        pct = 100 * count / len(signals)
        logger.info(f"   {sig}: {count:,} ({pct:.2f}%)")
    
    # High-confidence signals
    high_conf_long = ((pd.Series(signals) == 'LONG') & (pd.Series(confidence) >= confidence_threshold)).sum()
    high_conf_short = ((pd.Series(signals) == 'SHORT') & (pd.Series(confidence) >= confidence_threshold)).sum()
    logger.info(f"\n🎯 High-Confidence Signals (>={confidence_threshold:.0%}):")
    logger.info(f"   LONG:  {high_conf_long:,}")
    logger.info(f"   SHORT: {high_conf_short:,}")
    logger.info(f"   Total: {high_conf_long + high_conf_short:,}")
    
    # Initialize backtest
    logger.info("\n💼 Running backtest simulation...")
    capital = initial_capital
    position = None
    equity_curve = [capital]
    trades = []
    
    # Sequential processing - NO LOOK-AHEAD BIAS
    for i in range(len(test_df)):
        current_price = test_df.iloc[i]['close']
        current_time = test_df.iloc[i]['time']
        signal = signals[i]
        conf = confidence[i]
        
        # Check existing position
        if position is not None:
            # Calculate P&L
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            # Exit conditions
            should_exit = (
                pnl_pct >= profit_target or  # Profit target hit
                pnl_pct <= -stop_loss or     # Stop loss hit
                (position['side'] == 'LONG' and signal == 'SHORT' and conf >= confidence_threshold) or  # Reverse signal
                (position['side'] == 'SHORT' and signal == 'LONG' and conf >= confidence_threshold)
            )
            
            if should_exit:
                # Close position
                pnl = position['size'] * pnl_pct
                capital += pnl
                
                # Record trade
                exit_reason = 'PROFIT_TARGET' if pnl_pct >= profit_target else ('STOP_LOSS' if pnl_pct <= -stop_loss else 'REVERSE_SIGNAL')
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'holding_bars': i - position['entry_idx']
                })
                
                position = None
        
        # Enter new position (only if high confidence)
        if position is None and signal in ['LONG', 'SHORT'] and conf >= confidence_threshold:
            # Position sizing: 8% of current capital
            position_size = capital * 0.08
            
            position = {
                'side': signal,
                'entry_price': current_price,
                'entry_time': current_time,
                'entry_idx': i,
                'size': position_size,
                'confidence': conf
            }
        
        # Record equity
        equity_curve.append(capital)
    
    # Close any remaining position at the end
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
            'exit_reason': 'END_OF_BACKTEST',
            'holding_bars': len(test_df) - position['entry_idx']
        })
        
        position = None
    
    equity_curve.append(capital)
    
    # Calculate metrics
    logger.info("\n📊 Calculating performance metrics...")
    
    total_return = (capital - initial_capital) / initial_capital
    
    # Trade statistics
    if len(trades) > 0:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        
        total_wins = sum([t['pnl'] for t in wins])
        total_losses = abs(sum([t['pnl'] for t in losses]))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe Ratio
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(365 * 24 * 4)) if returns.std() > 0 else 0
        
        # Max Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
    else:
        win_rate = avg_win = avg_loss = profit_factor = sharpe_ratio = max_drawdown = 0
        wins = losses = []
    
    # Print results
    logger.info("")
    logger.info("="*80)
    logger.info("📊 BACKTEST RESULTS - 2025 (JAN-NOV)")
    logger.info("="*80)
    logger.info(f"   💰 Initial Capital:      ${initial_capital:,.2f}")
    logger.info(f"   💰 Final Capital:        ${capital:,.2f}")
    logger.info(f"   📈 Total Return:         {total_return:.2%}")
    logger.info(f"   📊 Sharpe Ratio:         {sharpe_ratio:.3f}")
    logger.info(f"   📉 Max Drawdown:         {max_drawdown:.2%}")
    logger.info(f"   ✅ Win Rate:             {win_rate:.2%}")
    logger.info(f"   🔢 Total Trades:         {len(trades)}")
    logger.info(f"   ✅ Winning Trades:       {len(wins)}")
    logger.info(f"   ❌ Losing Trades:        {len(losses)}")
    logger.info(f"   💵 Average Win:          ${avg_win:.2f}")
    logger.info(f"   💸 Average Loss:         ${avg_loss:.2f}")
    logger.info(f"   📊 Profit Factor:        {profit_factor:.2f}")
    logger.info("="*80)
    
    # Exit reason analysis
    if len(trades) > 0:
        exit_reasons = pd.Series([t['exit_reason'] for t in trades]).value_counts()
        logger.info("\n📊 Exit Reason Analysis:")
        for reason, count in exit_reasons.items():
            logger.info(f"   {reason}: {count} ({100*count/len(trades):.1f}%)")
    
    # Return results
    results = {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'total_trades': len(trades),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'average_win': avg_win,
        'average_loss': avg_loss,
        'profit_factor': profit_factor,
        'equity_curve': equity_curve,
        'trades': trades,
        'signal_distribution': signal_counts.to_dict(),
        'high_confidence_signals': {
            'LONG': int(high_conf_long),
            'SHORT': int(high_conf_short),
            'total': int(high_conf_long + high_conf_short)
        }
    }
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='2025년 완벽한 백테스트 (2019-2022 학습)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--profit', type=float, default=0.015, help='Profit target (default: 1.5%)')
    parser.add_argument('--stoploss', type=float, default=0.005, help='Stop loss (default: 0.5%)')
    parser.add_argument('--confidence', type=float, default=0.50, help='Confidence threshold (default: 0.50)')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("🚀 2025년 완벽한 백테스트 시작")
    logger.info("📚 학습: 2019-2022 | 🧪 테스트: 2025 (1월-11월)")
    logger.info("🤖 앙상블 모델: XGBoost + TabNet + CatBoost")
    logger.info("="*80)
    logger.info(f"💰 초기 자본:         ${args.capital:,.2f}")
    logger.info(f"📈 수익 목표:         {args.profit:.2%}")
    logger.info(f"📉 손절 기준:         {args.stoploss:.2%}")
    logger.info(f"🎯 신뢰도 임계값:      {args.confidence:.2%}")
    logger.info("="*80)
    
    start_time = datetime.now()
    
    try:
        # 1. Load data
        train_df = load_training_data()
        test_df = load_test_data_2025()
        
        # 2. Initialize components
        indicator_calculator = CustomIndicators()
        labeler = TripleBarrierLabeler(
            profit_target=args.profit,
            stop_loss=args.stoploss,
            time_limit_minutes=60
        )
        
        # 3. Process training data
        logger.info("\n" + "="*80)
        logger.info("📚 PROCESSING TRAINING DATA (2019-2022)")
        logger.info("="*80)
        train_df = process_data(train_df, indicator_calculator, labeler, is_training=True)
        
        # 4. Process test data (NO LABELS)
        logger.info("\n" + "="*80)
        logger.info("🧪 PROCESSING TEST DATA (2025)")
        logger.info("="*80)
        test_df = process_data(test_df, indicator_calculator, labeler, is_training=False)
        
        # 5. Train model
        signal_generator, available_features, train_metrics = train_ensemble_model(
            train_df, 
            indicator_calculator,
            confidence_threshold=args.confidence
        )
        
        # 6. Backtest on 2025
        backtest_results = backtest_2025(
            test_df,
            signal_generator,
            available_features,
            initial_capital=args.capital,
            profit_target=args.profit,
            stop_loss=args.stoploss,
            confidence_threshold=args.confidence
        )
        
        # 7. Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = project_root / "backtest_results"
        results_dir.mkdir(exist_ok=True)
        results_file = results_dir / f"2025_backtest_perfect_{timestamp}.json"
        
        # Prepare serializable results
        def make_serializable(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        results = {
            'backtest_info': {
                'train_period': {
                    'start': train_df['time'].min().isoformat(),
                    'end': train_df['time'].max().isoformat(),
                    'samples': len(train_df)
                },
                'test_period': {
                    'start': test_df['time'].min().isoformat(),
                    'end': test_df['time'].max().isoformat(),
                    'samples': len(test_df)
                },
                'timestamp': timestamp,
                'parameters': {
                    'initial_capital': args.capital,
                    'profit_target': args.profit,
                    'stop_loss': args.stoploss,
                    'confidence_threshold': args.confidence
                }
            },
            'model_performance': {
                'accuracy': float(train_metrics.get('accuracy', 0)),
                'f1_score': float(train_metrics.get('f1_score', 0)),
                'individual_models': {
                    name: {
                        'accuracy': float(metrics.get('accuracy', 0)),
                        'f1_score': float(metrics.get('f1_score', 0))
                    }
                    for name, metrics in train_metrics.get('individual_metrics', {}).items()
                }
            },
            'backtest_results': make_serializable(backtest_results)
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("✅ 백테스트 완료!")
        logger.info("="*80)
        logger.info(f"📄 결과 저장: {results_file}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"⏱️  실행 시간: {duration:.1f}초 ({duration/60:.1f}분)")
        logger.info("="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 백테스트 실패: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
