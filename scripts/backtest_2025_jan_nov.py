"""
2025년 1월~11월 백테스팅
=======================

2019-2022 학습 모델로 2025년 1-11월 데이터 백테스팅

사용법:
    python scripts/backtest_2025_jan_nov.py
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
from src.models.layer3_signal.signal_generator import SignalGenerator
from src.utils.logger import get_logger

logger = get_logger("backtest_2025")


def load_data():
    """데이터 로드"""
    logger.info("\n" + "=" * 80)
    logger.info("📂 데이터 로딩")
    logger.info("=" * 80)
    
    # 학습 데이터 (2019-2022)
    train_file = Path("data/raw/BTCUSDT_15m_2019_2024_full.csv")
    
    if not train_file.exists():
        raise FileNotFoundError(f"학습 데이터 파일 없음: {train_file}")
    
    logger.info(f"학습 데이터 로드: {train_file}")
    train_df = pd.read_csv(train_file)
    train_df['time'] = pd.to_datetime(train_df['time'])
    
    # 2019-2022만 필터링
    train_start = datetime(2019, 1, 1)
    train_end = datetime(2022, 12, 31)
    train_df = train_df[(train_df['time'] >= train_start) & (train_df['time'] <= train_end)].copy()
    
    # 테스트 데이터 (2025년 1-11월)
    test_file = Path("data/raw/BTCUSDT_15m_2025_jan_nov.csv")
    
    if not test_file.exists():
        raise FileNotFoundError(f"테스트 데이터 파일 없음: {test_file}")
    
    logger.info(f"테스트 데이터 로드: {test_file}")
    test_df = pd.read_csv(test_file)
    test_df['time'] = pd.to_datetime(test_df['time'])
    
    logger.info(f"")
    logger.info(f"✅ 데이터 로드 완료")
    logger.info(f"   📚 학습:  {len(train_df):,}개 캔들 ({train_df['time'].min()} ~ {train_df['time'].max()})")
    logger.info(f"   🧪 테스트: {len(test_df):,}개 캔들 ({test_df['time'].min()} ~ {test_df['time'].max()})")
    logger.info(f"   💰 가격 범위 (학습): ${train_df['close'].min():,.0f} ~ ${train_df['close'].max():,.0f}")
    logger.info(f"   💰 가격 범위 (테스트): ${test_df['close'].min():,.0f} ~ ${test_df['close'].max():,.0f}")
    
    return train_df, test_df


def process_data(df, indicator_calculator, labeler, name="data"):
    """지표 계산 및 라벨 생성"""
    logger.info(f"\n🔧 {name} 처리 중...")
    
    # 지표 계산
    df = indicator_calculator.calculate_all_indicators(df)
    logger.info(f"   ✅ 지표 계산 완료 ({len(df):,}개 샘플)")
    
    # 라벨 생성
    df = labeler.create_labels(df)
    
    # 라벨 통계
    stats = labeler.get_label_statistics(df)
    logger.info(f"   ✅ 라벨 생성 완료 ({stats['total_samples']:,}개 샘플)")
    logger.info(f"      LONG: {stats['label_counts']['LONG']:,}개 ({stats['label_percentages']['LONG']:.1f}%)")
    logger.info(f"      SHORT: {stats['label_counts']['SHORT']:,}개 ({stats['label_percentages']['SHORT']:.1f}%)")
    logger.info(f"      NEUTRAL: {stats['label_counts']['NEUTRAL']:,}개 ({stats['label_percentages']['NEUTRAL']:.1f}%)")
    
    return df


def train_model(train_df, signal_generator):
    """모델 학습"""
    logger.info("\n" + "=" * 80)
    logger.info("📚 2019-2022 데이터로 모델 학습")
    logger.info("=" * 80)
    logger.info(f"학습 샘플 수: {len(train_df):,}개")
    
    metrics = signal_generator.train(
        train_df,
        label_col='tb_label',
        test_size=0.2,
        balance_method='class_weight'
    )
    
    logger.info("")
    logger.info("✅ 학습 완료")
    logger.info(f"   📊 정확도:  {metrics['accuracy']:.4f}")
    logger.info(f"   📊 F1 점수: {metrics['f1_score']:.4f}")
    
    return metrics


def backtest_continuous(test_df, signal_generator, indicator_calculator, initial_capital=10000.0, 
                       profit_target=0.015, stop_loss=0.005):
    """연속 백테스팅"""
    logger.info("\n" + "=" * 80)
    logger.info("🧪 2025년 1-11월 백테스팅")
    logger.info("=" * 80)
    
    # 특성 컬럼 가져오기
    feature_cols = indicator_calculator.get_feature_names()
    available_features = [col for col in feature_cols if col in test_df.columns]
    
    # 신호 생성
    logger.info("신호 생성 중...")
    signals, confidence, _ = signal_generator.predict(test_df[available_features])
    
    # 백테스팅 실행
    logger.info("백테스팅 실행 중...")
    capital = initial_capital
    position = None
    
    equity_curve = [capital]
    trades = []
    
    for i in range(len(test_df)):
        current_price = test_df.iloc[i]['close']
        current_time = test_df.iloc[i]['time']
        signal = signals[i]
        conf = confidence[i]
        
        # 포지션 보유 중
        if position is not None:
            # 손익 계산
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            # 청산 조건
            should_exit = (
                pnl_pct >= profit_target or
                pnl_pct <= -stop_loss or
                (position['side'] == 'LONG' and signal == 'SHORT') or
                (position['side'] == 'SHORT' and signal == 'LONG')
            )
            
            if should_exit:
                # 포지션 청산
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
        
        # 신규 포지션 진입
        if position is None and signal in ['LONG', 'SHORT'] and conf >= 0.65:
            # 포지션 크기: 자본의 8%
            position_size = capital * 0.08
            
            position = {
                'side': signal,
                'entry_price': current_price,
                'entry_time': current_time,
                'size': position_size
            }
        
        equity_curve.append(capital)
    
    # 남은 포지션 청산
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
    
    # 지표 계산
    total_return = (capital - initial_capital) / initial_capital
    
    if len(trades) > 0:
        returns = [t['pnl'] / initial_capital for t in trades]
        win_trades = [t for t in trades if t['pnl'] > 0]
        lose_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(win_trades) / len(trades)
        
        # 샤프 비율
        if np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 최대 낙폭
        peak = initial_capital
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        
        # 평균 손익
        avg_win = np.mean([t['pnl'] for t in win_trades]) if len(win_trades) > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in lose_trades]) if len(lose_trades) > 0 else 0
        
        # 손익비
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
    
    # 결과 출력
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 백테스팅 결과")
    logger.info("=" * 80)
    logger.info(f"   💰 초기 자본:        ${initial_capital:,.2f}")
    logger.info(f"   💰 최종 자본:        ${capital:,.2f}")
    logger.info(f"   📈 총 수익률:        {total_return:.2%}")
    logger.info(f"   📊 샤프 비율:        {sharpe_ratio:.3f}")
    logger.info(f"   📉 최대 낙폭:        {max_dd:.2%}")
    logger.info(f"   ✅ 승률:            {win_rate:.2%}")
    logger.info(f"   🔢 총 거래:          {len(trades)}회")
    logger.info(f"   ✅ 승리 거래:        {len(win_trades)}회")
    logger.info(f"   ❌ 손실 거래:        {len(lose_trades)}회")
    logger.info(f"   💵 평균 수익:        ${avg_win:,.2f}")
    logger.info(f"   💸 평균 손실:        ${avg_loss:,.2f}")
    logger.info(f"   📊 손익비:           {profit_factor:.2f}")
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
    """메인 함수"""
    parser = argparse.ArgumentParser(description='2025년 1-11월 백테스팅')
    parser.add_argument('--capital', type=float, default=10000, help='초기 자본')
    parser.add_argument('--profit', type=float, default=0.015, help='익절 목표 (기본: 1.5%)')
    parser.add_argument('--stoploss', type=float, default=0.005, help='손절 (기본: 0.5%)')
    
    args = parser.parse_args()
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 2025년 1-11월 백테스팅 (2019-2022 학습 모델)")
    logger.info("=" * 80)
    logger.info(f"초기 자본: ${args.capital:,.2f}")
    logger.info(f"익절 목표: {args.profit:.2%}")
    logger.info(f"손절:     {args.stoploss:.2%}")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # 데이터 로드
        train_df, test_df = load_data()
        
        # 컴포넌트 초기화
        indicator_calculator = CustomIndicators()
        labeler = TripleBarrierLabeler(
            profit_target=args.profit,
            stop_loss=args.stoploss,
            time_limit_minutes=60
        )
        signal_generator = SignalGenerator(model_type='xgboost')
        
        # 학습 데이터 처리
        logger.info("\n" + "=" * 80)
        logger.info("📚 학습 데이터 처리")
        logger.info("=" * 80)
        train_df = process_data(train_df, indicator_calculator, labeler, "학습 데이터")
        
        # 테스트 데이터 처리
        logger.info("\n" + "=" * 80)
        logger.info("🧪 테스트 데이터 처리")
        logger.info("=" * 80)
        test_df = process_data(test_df, indicator_calculator, labeler, "테스트 데이터")
        
        # 모델 학습
        train_metrics = train_model(train_df, signal_generator)
        
        # 백테스팅
        backtest_results = backtest_continuous(
            test_df, 
            signal_generator, 
            indicator_calculator,
            initial_capital=args.capital,
            profit_target=args.profit,
            stop_loss=args.stoploss
        )
        
        # 결과 저장
        output_dir = Path('backtest_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f'backtest_2025_jan_nov_{timestamp}.json'
        
        # 학습 지표에서 모델/DataFrame 제거
        train_metrics_filtered = {
            k: v for k, v in train_metrics.items() 
            if not isinstance(v, (pd.DataFrame, object)) or isinstance(v, (int, float, str, list, dict))
        }
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
                'start': '2025-01-01',
                'end': '2025-11-30',
                'samples': len(test_df)
            },
            'backtest_results': backtest_results,
            'parameters': {
                'initial_capital': args.capital,
                'profit_target': args.profit,
                'stop_loss': args.stoploss
            }
        }
        
        # JSON 직렬화
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
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ 완료")
        logger.info("=" * 80)
        logger.info(f"📄 결과 저장: {results_file}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"⏱️  소요 시간: {duration:.1f}초 ({duration/60:.1f}분)")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 실패: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
