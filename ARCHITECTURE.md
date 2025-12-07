# 🏗️ System Architecture

## Overview

This is a production-grade, AI-powered cryptocurrency futures trading system built on a revolutionary **4-Layer Architecture** that prioritizes **response over prediction**.

## Core Philosophy

### ❌ Traditional Approach
"Where will BTC be tomorrow?" → **Predicting the future**

### ✅ Our Approach
"Is taking a LONG position statistically advantageous right now?" → **Responding optimally to the present**

## 4-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                  INPUT: Market Data Stream                      │
│         (OHLCV, Funding Rate, Order Book, Trades)              │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│           LAYER 1: Market Regime Classifier                     │
│                                                                 │
│  Model: LightGBM (Phase 1) → TFT (Phase 2)                    │
│  Role: Identify current market state                           │
│  Output: 7 market regimes                                      │
│    • TRENDING_UP       • TRENDING_DOWN                         │
│    • RANGING           • HIGH_VOLATILITY                       │
│    • LOW_VOLATILITY    • BREAKOUT                              │
│    • REVERSAL                                                  │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│           LAYER 2: Pattern Recognizer                           │
│                                                                 │
│  Model: TSMixer + PatchTST Ensemble (Phase 2)                 │
│  Role: Extract complex patterns and long-term dependencies     │
│  Output: High-dimensional pattern feature vectors              │
│  Strength: Captures subtle patterns invisible to humans        │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│           LAYER 3: Signal Generator ⭐ CORE                      │
│                                                                 │
│  Model: XGBoost (Phase 1) → TabNet + FT-Transformer (Phase 2) │
│  Role: Generate trading signals                                │
│  Training: Triple Barrier Method labels                        │
│  Output:                                                       │
│    • Signal: LONG / SHORT / NEUTRAL                            │
│    • Confidence: 0-100%                                        │
│    • Probabilities: [P(Long), P(Short), P(Neutral)]           │
│                                                                 │
│  Decision Rule:                                                │
│    IF confidence ≥ 65% → Execute signal                        │
│    ELSE → Stay NEUTRAL (do nothing)                            │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│           LAYER 4: Position Manager                             │
│                                                                 │
│  Model: Rule-Based (Phase 1) → SAC RL (Phase 2)               │
│  Role: Complex position management                             │
│  Actions:                                                      │
│    • Position sizing (dynamic based on confidence)             │
│    • Partial profit taking                                     │
│    • Adding to positions                                       │
│    • Dynamic stop loss adjustment                              │
│    • Emergency exits                                           │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│              RISK MANAGEMENT (3-Tier Safety Net)                │
│                                                                 │
│  Level 1 - Position:                                           │
│    • Max 8% of capital per position                            │
│    • Max 5x leverage                                           │
│    • Stop loss: 0.5% | Take profit: 1.5%                      │
│                                                                 │
│  Level 2 - Daily:                                              │
│    • Max 2% daily loss → Halt trading                          │
│    • Max 15 trades per day                                     │
│    • Max 8 consecutive losses → Halt                           │
│                                                                 │
│  Level 3 - System (EMERGENCY):                                 │
│    • Max 5% drawdown → SHUTDOWN                                │
│    • Volatility 3x normal → Close all positions                │
│    • Sharpe < 0.5 → Review strategy                            │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                  OUTPUT: Trade Execution                        │
│              (Binance Futures API via CCXT)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Triple Barrier Method

The core innovation that makes this system work.

### Traditional Labeling ❌
"Will price go up?" → Binary classification based on future price

### Triple Barrier Labeling ✅
"What action should I take?" → Multi-class based on profit/loss outcomes

```
Entry Price: $45,000
    │
    ├─ Upper Barrier: $45,675 (+1.5%) → Label: LONG
    │
    ├─ Lower Barrier: $44,775 (-0.5%) → Label: SHORT
    │
    └─ Time Barrier: 60 minutes → Label: NEUTRAL
```

**Example:**
1. Enter at $45,000
2. Price hits $45,675 after 20 minutes
3. → Label this timestamp as **LONG** (profit target hit)

This creates labels that directly correspond to profitable actions!

## Data Flow

### 1. Data Collection
```
Binance API (WebSocket + REST)
    ↓
Raw Data (OHLCV, Funding Rate, Order Book, Trades)
    ↓
TimescaleDB (High-frequency time-series storage)
```

### 2. Data Processing
```
Raw Data
    ↓
Normalization (Price → Percentage changes)
    ↓
Feature Engineering (100+ indicators)
    ↓
Triple Barrier Labeling
    ↓
Training Dataset
```

### 3. Model Training
```
Training Data (Features + Labels)
    ↓
Walk-Forward Validation
    ├─ Train: 6 months
    ├─ Test: 1 month
    └─ Step: 1 month forward
    ↓
Trained Models (saved to disk)
```

### 4. Live Trading
```
Real-time Market Data
    ↓
Feature Calculation
    ↓
Layer 1: Regime Classification
    ↓
Layer 2: Pattern Recognition
    ↓
Layer 3: Signal Generation (with confidence)
    ↓
Risk Management Checks
    ↓
Layer 4: Position Management
    ↓
Order Execution
```

## Technology Stack

### Core Framework
- **Python 3.11**: Main language
- **FastAPI**: REST API server
- **Pydantic**: Configuration management

### Machine Learning
- **XGBoost**: Signal generation (Phase 1)
- **LightGBM**: Regime classification (Phase 1)
- **PyTorch**: Deep learning models (Phase 2)
- **scikit-learn**: Preprocessing & metrics
- **NumPy/Pandas**: Data manipulation

### Database & Storage
- **PostgreSQL + TimescaleDB**: Time-series data
- **Redis**: Real-time caching & pub/sub
- **SQLAlchemy**: ORM

### Trading Infrastructure
- **CCXT**: Unified exchange API
- **python-binance**: Binance-specific optimizations
- **WebSocket**: Real-time data streaming

### Monitoring & Logging
- **Loguru**: Structured logging
- **TensorBoard**: Model training visualization
- **Dash/Plotly**: Trading dashboard
- **Prometheus**: Metrics collection

### DevOps
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration
- **Git**: Version control

## Directory Structure

```
webapp/
├── src/                          # Source code
│   ├── data_collection/          # Binance data pipeline
│   │   ├── binance_client.py    # CCXT wrapper
│   │   ├── ohlcv_collector.py   # Candlestick data
│   │   ├── funding_collector.py # Funding rates
│   │   └── orderbook_collector.py
│   │
│   ├── data_processing/          # Feature engineering
│   │   ├── normalizer.py        # Price normalization
│   │   ├── feature_engineer.py  # 100+ indicators
│   │   └── triple_barrier.py    # Action-based labels
│   │
│   ├── models/                   # 4-Layer AI models
│   │   ├── layer1_regime/       # Market regime
│   │   ├── layer2_pattern/      # Pattern recognition
│   │   ├── layer3_signal/       # Signal generation ⭐
│   │   └── layer4_position/     # Position management
│   │
│   ├── backtesting/              # Walk-Forward validation
│   │   ├── engine.py            # Backtest engine
│   │   ├── walk_forward.py      # WF validation
│   │   └── metrics.py           # Performance metrics
│   │
│   ├── risk_management/          # 3-Tier safety
│   │   └── risk_manager.py      # Comprehensive risk system
│   │
│   ├── trading/                  # Live execution
│   │   ├── executor.py          # Order execution
│   │   ├── position_tracker.py  # Position tracking
│   │   └── paper_trading.py     # Testnet trading
│   │
│   ├── utils/                    # Utilities
│   │   ├── config.py            # Configuration
│   │   ├── logger.py            # Logging
│   │   └── database.py          # DB connections
│   │
│   └── main.py                   # Application entry
│
├── config/                       # Configuration files
├── data/                         # Data storage
│   ├── raw/                     # Raw market data
│   ├── processed/               # Processed features
│   └── models/                  # Trained models
│
├── tests/                        # Unit & integration tests
├── scripts/                      # Utility scripts
├── logs/                         # Application logs
├── notebooks/                    # Jupyter analysis
│
├── docker-compose.yml            # Service orchestration
├── Dockerfile                    # Application container
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

## Deployment Options

### Option 1: Docker Compose (Recommended)
```bash
docker-compose up -d
```
- ✅ Easy setup
- ✅ Isolated environment
- ✅ All services managed together

### Option 2: Manual Deployment
```bash
python -m src.main
```
- ✅ More control
- ✅ Easier debugging
- ⚠️ Requires manual service setup

### Option 3: Kubernetes (Production)
```bash
kubectl apply -f k8s/
```
- ✅ High availability
- ✅ Auto-scaling
- ✅ Production-grade
- ⚠️ Complex setup

## Performance Expectations

### Phase 1 (Current - Stable Foundation)
- **Target Return**: 10-15% annualized
- **Sharpe Ratio**: 0.8-1.2
- **Max Drawdown**: <8%
- **Win Rate**: >52%
- **Primary Goal**: System stability, loss prevention

### Phase 2 (3-6 months - Optimization)
- **Target Return**: 15-25% annualized
- **Sharpe Ratio**: 1.2-1.8
- **Max Drawdown**: <6%
- **Win Rate**: >58%
- **Primary Goal**: Performance enhancement

### Long-term (Sustainable)
- **Target Return**: 20-30% annualized
- **Note**: This is top 10% of professional hedge funds

## Key Success Factors

### 1. Data Quality > Model Complexity
Clean, accurate data beats fancy algorithms every time.

### 2. Overfitting Prevention
Walk-Forward validation ensures models work on unseen data.

### 3. Risk Management
Survival first, profits second. Always.

### 4. Gradual Improvement
Don't try to build everything at once. Iterate.

### 5. Realistic Expectations
Sustainable 20-30% annual returns beat risky 100%+ claims.

## Monitoring & Alerts

### Real-time Monitoring
- Position status
- Current P&L
- Risk metrics
- System health

### Alert Triggers
- Position opened/closed
- Daily loss approaching limit
- System drawdown warning
- Model performance degradation
- API connection issues

### Notification Channels
- Telegram bot
- Discord webhook
- Email alerts
- SMS (optional)

## Scaling Strategy

### Phase 1: Single Symbol
- Start with BTC/USDT only
- Perfect the core system
- Build confidence

### Phase 2: Multi-Symbol
- Add ETH/USDT, BNB/USDT
- Diversification benefits
- Still manageable

### Phase 3: Portfolio
- 5-10 major cryptocurrencies
- Correlation analysis
- Portfolio optimization

## Security Considerations

### API Keys
- Never commit to version control
- Use environment variables
- Enable IP whitelist on Binance
- Separate testnet and mainnet keys

### Database
- Strong passwords
- Limited external access
- Regular backups
- Encrypted connections

### Code
- No hardcoded secrets
- Input validation
- Rate limiting
- Error handling

## Testing Strategy

### Unit Tests
- Individual component testing
- 80%+ code coverage target

### Integration Tests
- End-to-end workflows
- Database interactions
- API communications

### Backtesting
- Historical data validation
- Walk-Forward testing
- Out-of-sample verification

### Paper Trading
- Minimum 3 months
- Real market conditions
- Zero financial risk

## Future Enhancements

### Phase 2 Model Upgrades
- **Layer 1**: TFT (Temporal Fusion Transformer)
- **Layer 2**: TSMixer + PatchTST ensemble
- **Layer 3**: TabNet + FT-Transformer + CatBoost
- **Layer 4**: SAC (Soft Actor-Critic) RL agent

### Additional Features
- Sentiment analysis (Twitter, Reddit, news)
- Multi-timeframe ensemble
- Adversarial validation
- Meta-learning for rapid adaptation
- Multi-exchange arbitrage

### Infrastructure
- Kubernetes deployment
- High-availability setup
- Automatic failover
- Distributed training
- GPU acceleration

---

**This architecture is designed for one thing: Sustainable, long-term profitability with minimal risk.**

The key is not predicting what will happen, but knowing what to do when it does happen.
