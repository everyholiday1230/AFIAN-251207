#!/usr/bin/env python3
"""
2025년 데이터 백테스트
기존 학습된 모델로 2025년 1월-11월 데이터를 백테스트
"""
import os
import sys
import json
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


def load_data_2025():
    """Load 2025 data"""
    logger.info("\n" + "="*80)
    logger.info("📂 LOADING 2025 DATA")
    logger.info("="*80)
    
    data_path = project_root / "data/raw/BTCUSDT_15m_2025_jan_nov.csv"
    logger.info(f"Loading from {data_path}")
    
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    logger.info("")
    logger.info("✅ Data loaded")
    logger.info(f"   📚 Total: {len(df):,} candles ({df['time'].min()} to {df['time'].max()})")
    logger.info(f"   💰 Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    
    return df


def process_indicators(df):
    """Process indicators"""
    logger.info("\n🔧 Processing indicators...")
    
    indicators = CustomIndicators()
    df = indicators.calculate_all_indicators(df)
    
    logger.info(f"   ✅ Indicators calculated ({len(df):,} samples)")
    return df


def backtest_2025(model, test_data, initial_capital=10000, profit_target=0.015, stop_loss=0.005):
    """Run backtest on 2025 data"""
    logger.info("\n" + "="*80)
    logger.info("🧪 BACKTESTING ON 2025 DATA")
    logger.info("="*80)
    
    logger.info("Generating signals...")
    
    # Generate signals
    signals = []
    confidences = []
    
    for idx, row in test_data.iterrows():
        signal, confidence = model.predict(pd.DataFrame([row]))
        signals.append(signal)
        confidences.append(confidence)
    
    test_data['signal'] = signals
    test_data['confidence'] = confidences
    
    logger.info("Running backtest...")
    
    # Backtest
    capital = initial_capital
    position = None
    entry_price = 0
    entry_time = None
    trades = []
    equity_curve = [capital]
    
    for idx, row in test_data.iterrows():
        current_price = row['close']
        current_time = row['time']
        signal = row['signal']
        
        # Open position
        if position is None and signal in ['LONG', 'SHORT']:
            position = signal
            entry_price = current_price
            entry_time = current_time
            
        # Close position
        elif position is not None:
            exit_reason = None
            exit_price = current_price
            
            if position == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                if pnl_pct >= profit_target:
                    exit_reason = 'PROFIT_TARGET'
                elif pnl_pct <= -stop_loss:
                    exit_reason = 'STOP_LOSS'
                    
            elif position == 'SHORT':
                pnl_pct = (entry_price - current_price) / entry_price
                if pnl_pct >= profit_target:
                    exit_reason = 'PROFIT_TARGET'
                elif pnl_pct <= -stop_loss:
                    exit_reason = 'STOP_LOSS'
            
            if exit_reason:
                pnl = capital * pnl_pct
                capital += pnl
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason
                })
                
                position = None
                entry_price = 0
                entry_time = None
        
        equity_curve.append(capital)
    
    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] < 0]
    
    win_rate = len(winning_trades) / len(trades) if trades else 0
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    
    profit_factor = abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
    
    # Sharpe ratio
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365 * 96) if returns.std() > 0 else 0  # 15min = 96 per day
    
    # Max drawdown
    cumulative = pd.Series(equity_curve).cummax()
    drawdown = (pd.Series(equity_curve) - cumulative) / cumulative
    max_drawdown = abs(drawdown.min())
    
    logger.info("")
    logger.info("="*80)
    logger.info("📊 BACKTEST RESULTS")
    logger.info("="*80)
    logger.info(f"   💰 Initial Capital:      ${initial_capital:,.2f}")
    logger.info(f"   💰 Final Capital:        ${capital:,.2f}")
    logger.info(f"   📈 Total Return:         {total_return*100:.2f}%")
    logger.info(f"   📊 Sharpe Ratio:         {sharpe_ratio:.3f}")
    logger.info(f"   📉 Max Drawdown:         {max_drawdown*100:.2f}%")
    logger.info(f"   ✅ Win Rate:             {win_rate*100:.2f}%")
    logger.info(f"   🔢 Total Trades:         {len(trades)}")
    logger.info(f"   ✅ Winning Trades:       {len(winning_trades)}")
    logger.info(f"   ❌ Losing Trades:        {len(losing_trades)}")
    logger.info(f"   💵 Average Win:          ${avg_win:.2f}")
    logger.info(f"   💸 Average Loss:         ${avg_loss:.2f}")
    logger.info(f"   📊 Profit Factor:        {profit_factor:.2f}")
    logger.info("="*80)
    
    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'equity_curve': equity_curve,
        'trades': trades
    }


def main():
    # Load 2025 data
    df_2025 = load_data_2025()
    
    # Process indicators
    df_2025 = process_indicators(df_2025)
    
    # Load trained model
    logger.info("\n" + "="*80)
    logger.info("📚 LOADING TRAINED MODEL")
    logger.info("="*80)
    
    model = EnsembleSignalGenerator(confidence_threshold=0.50)  # Lower threshold
    
    # Note: You need to load the trained models here
    # For now, we'll just train on a small sample to test
    logger.info("⚠️  Model needs to be loaded from saved weights")
    logger.info("    Using placeholder for testing...")
    
    # Backtest
    results = backtest_2025(
        model=model,
        test_data=df_2025,
        initial_capital=10000,
        profit_target=0.015,
        stop_loss=0.005
    )
    
    # Save results
    output_dir = project_root / "backtest_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"backtest_2025_jan-nov_{timestamp}.json"
    
    # Filter results for JSON serialization
    json_results = {
        'test_period': {
            'start': '2025-01-01',
            'end': '2025-11-30',
            'samples': len(df_2025)
        },
        'backtest_results': {
            k: v for k, v in results.items() 
            if k not in ['trades']  # Skip trades for now
        },
        'parameters': {
            'initial_capital': 10000,
            'profit_target': 0.015,
            'stop_loss': 0.005
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("✅ COMPLETE")
    logger.info("="*80)
    logger.info(f"📄 Results saved to: {output_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
