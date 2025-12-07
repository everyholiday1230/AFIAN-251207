# 🚀 실행 가이드

## 📋 현재 시스템 파라미터

### Triple Barrier 설정
```python
PROFIT_TARGET = 0.015    # 1.5% 익절
STOP_LOSS = 0.005        # 0.5% 손절
TIME_LIMIT = 60분        # 최대 보유 시간
```

### 백테스팅 설정
```python
SYMBOL = 'BTC/USDT'
TIMEFRAME = '15m'        # 15분봉
START_DATE = '2023-01-01'
END_DATE = '2024-12-31'
TRAIN_WINDOW = 180일     # 6개월 학습
TEST_WINDOW = 30일       # 1개월 테스트
STEP = 30일              # Walk-Forward 이동
```

### 초기 자본
```python
INITIAL_CAPITAL = $10,000
```

---

## 🎯 파라미터 변경 방법

### 옵션 1: 커맨드 라인에서 변경
```bash
python scripts/train_and_backtest.py \
  --symbol BTC/USDT \
  --timeframe 15m \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --capital 10000
```

### 옵션 2: 코드에서 직접 변경

`scripts/train_and_backtest.py` 파일 열기:

```python
# Line 39-58: 파라미터 변경
def __init__(
    self,
    symbol: str = 'BTC/USDT',           # 여기 변경
    timeframe: str = '15m',             # 여기 변경
    start_date: str = '2023-01-01',     # 여기 변경
    end_date: str = '2024-12-31',       # 여기 변경
    initial_capital: float = 10000.0,   # 여기 변경
    # Triple Barrier parameters
    profit_target: float = 0.015,       # 여기 변경
    stop_loss: float = 0.005,           # 여기 변경
    time_limit_minutes: int = 60,       # 여기 변경
    # Walk-Forward parameters
    train_window_days: int = 180,       # 여기 변경
    test_window_days: int = 30,         # 여기 변경
    step_days: int = 30,                # 여기 변경
):
```

### 옵션 3: Custom Indicator 파라미터 변경

`src/data_processing/custom_indicators.py` 파일:

```python
# Line 55-70: 지표 파라미터
def __init__(
    self,
    # AI Oscillator parameters
    rsi_period: int = 60,               # 여기 변경
    rsi_lookback: int = 300,            # 여기 변경
    stoch_length: int = 60,             # 여기 변경
    smooth_k: int = 9,                  # 여기 변경
    smooth_d: int = 5,                  # 여기 변경
    stoch_lookback: int = 240,          # 여기 변경
    mfi_length: int = 60,               # 여기 변경
    velocity_period: int = 1,           # 여기 변경
    volatility_period: int = 20,        # 여기 변경
    trend_period: int = 10,             # 여기 변경
    # Impulse MACD parameters
    length_ma: int = 34,                # 여기 변경
    length_signal: int = 9              # 여기 변경
):
```

---

## 🏃 실행 방법

### 1. 환경 설정 (최초 1회만)
```bash
# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 데이터베이스 설정 (선택)
# docker-compose up -d timescaledb redis
```

### 2. 학습 및 백테스팅 실행

#### 기본 실행 (기본 파라미터 사용)
```bash
python scripts/train_and_backtest.py
```

#### 커스텀 파라미터로 실행
```bash
python scripts/train_and_backtest.py \
  --symbol ETH/USDT \
  --timeframe 30m \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --capital 50000
```

### 3. 실행 중 로그 확인
```bash
# 별도 터미널에서
tail -f logs/trading.log
```

---

## 📊 결과 확인

### 백테스트 결과 위치
```
backtest_results/
└── backtest_YYYYMMDD_HHMMSS.json
```

### 결과 파일 내용
```json
{
  "fold_results": [
    {
      "fold": 1,
      "total_return": 0.0856,
      "sharpe_ratio": 1.234,
      "max_drawdown": 0.0432,
      "win_rate": 0.567,
      "total_trades": 45
    }
  ],
  "summary": {
    "avg_return_per_fold": 0.0723,
    "avg_sharpe": 1.156,
    "avg_win_rate": 0.543
  }
}
```

### 결과 분석 스크립트 (예정)
```bash
python scripts/analyze_results.py backtest_results/backtest_20241207_120000.json
```

---

## ⚡ 빠른 테스트 (소량 데이터)

짧은 기간으로 시스템 테스트:
```bash
python scripts/train_and_backtest.py \
  --start 2024-10-01 \
  --end 2024-12-01 \
  --capital 1000
```

---

## 🔧 성능 최적화 팁

### 1. 더 긴 학습 기간
```python
train_window_days: int = 365  # 1년으로 증가
```

### 2. 더 짧은 테스트 기간
```python
test_window_days: int = 14    # 2주로 감소
```

### 3. Triple Barrier 조정

**더 보수적 (안전):**
```python
profit_target: float = 0.02   # 2%로 증가
stop_loss: float = 0.01       # 1%로 증가
```

**더 공격적:**
```python
profit_target: float = 0.01   # 1%로 감소
stop_loss: float = 0.003      # 0.3%로 감소
```

### 4. 다른 타임프레임 시도

**단기 (스캘핑):**
```python
timeframe: str = '5m'
time_limit_minutes: int = 30
```

**중기:**
```python
timeframe: str = '1h'
time_limit_minutes: int = 240  # 4시간
```

---

## 🐛 문제 해결

### 에러: "Binance API error"
```bash
# API 키 확인
echo $BINANCE_API_KEY

# 테스트넷 사용
BINANCE_TESTNET=true python scripts/train_and_backtest.py
```

### 에러: "Insufficient data"
```python
# 더 짧은 기간으로 시도
--start 2024-01-01 --end 2024-06-30
```

### 에러: "Memory error"
```python
# 더 짧은 lookback 기간
rsi_lookback: int = 150        # 300에서 감소
stoch_lookback: int = 120      # 240에서 감소
```

### 실행이 너무 느림
```python
# Walk-Forward step 증가
step_days: int = 60            # 30일에서 60일로
```

---

## 📈 권장 워크플로우

### 1단계: 짧은 기간 테스트
```bash
python scripts/train_and_backtest.py \
  --start 2024-10-01 \
  --end 2024-12-01
```
**목적**: 시스템이 정상 작동하는지 확인

### 2단계: 중간 기간 검증
```bash
python scripts/train_and_backtest.py \
  --start 2024-01-01 \
  --end 2024-12-31
```
**목적**: 1년 데이터로 성능 확인

### 3단계: 전체 기간 백테스팅
```bash
python scripts/train_and_backtest.py \
  --start 2023-01-01 \
  --end 2024-12-31
```
**목적**: 2년 데이터로 최종 검증

### 4단계: 파라미터 최적화
다양한 파라미터 조합 시도:
- Profit target: 1%, 1.5%, 2%
- Stop loss: 0.3%, 0.5%, 1%
- Time limit: 30분, 60분, 120분

### 5단계: 최고 성능 파라미터 선택
- Sharpe ratio가 가장 높은 조합
- Max drawdown이 가장 낮은 조합
- Win rate와 total return의 균형

---

## 🎯 성공 기준

### Phase 1 목표 (기본 검증)
- ✅ Sharpe Ratio > 0.5
- ✅ Max Drawdown < 10%
- ✅ Win Rate > 50%
- ✅ Positive return 60% 이상의 fold

### Phase 2 목표 (프로덕션 준비)
- ✅ Sharpe Ratio > 0.8
- ✅ Max Drawdown < 8%
- ✅ Win Rate > 52%
- ✅ Positive return 70% 이상의 fold

---

## 💡 다음 단계

백테스팅 성공 후:

1. **페이퍼 트레이딩 (3개월)**
   ```bash
   python -m src.trading.paper_trading
   ```

2. **성능 모니터링**
   - 일일 수익률 추적
   - Sharpe ratio 계산
   - Drawdown 모니터링

3. **실전 투자 (소액)**
   - 계좌의 1-2%로 시작
   - 레버리지 3배 이하
   - 매일 모니터링

---

## 🚨 중요 알림

### 절대 하지 말 것
❌ 백테스팅 없이 실전 투자  
❌ 리스크 한도 비활성화  
❌ 과도한 레버리지 사용  
❌ 감정적 개입  

### 반드시 할 것
✅ 최소 3개월 페이퍼 트레이딩  
✅ 매일 성능 체크  
✅ 리스크 한도 준수  
✅ 작은 금액으로 시작  

---

## 📞 도움말

### 로그 확인
```bash
# 전체 로그
cat logs/trading.log

# 에러만
grep ERROR logs/trading.log

# 최근 100줄
tail -100 logs/trading.log
```

### 시스템 상태 확인
```python
python -c "from src.utils.database import check_all_connections; print(check_all_connections())"
```

### 지표 테스트
```bash
python src/data_processing/custom_indicators.py
```

---

**준비되셨나요? 실행해봅시다! 🚀**

```bash
python scripts/train_and_backtest.py
```

행운을 빕니다! 💰📈
