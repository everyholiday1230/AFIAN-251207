# 🚀 Crypto Futures AI Trading System

**An Enterprise-Grade, 4-Layer AI-Powered Cryptocurrency Futures Automated Trading System**

> "예측이 아닌 대응" - We don't predict the future; we respond optimally to the present.

---

## 🎯 System Philosophy

### Paradigm Shift
- ❌ Traditional: "Where will BTC be tomorrow?"
- ✅ Our Approach: "Is taking a LONG position statistically advantageous right now?"

### Core Innovation
AI learns to select the **most profitable action** in the current market state, not to forecast future prices.

---

## 🏗️ 4-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 4: Position Manager                 │
│              (SAC/PPO - Complex Position Management)         │
│              부분익절, 추가진입, 동적 손절                    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                   Layer 3: Signal Generator                  │
│         (TabNet + FT-Transformer + CatBoost Ensemble)        │
│                 LONG/SHORT/NEUTRAL + Confidence              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                  Layer 2: Pattern Recognizer                 │
│              (TSMixer + PatchTST Ensemble)                   │
│            차트 패턴 및 장기 의존성 학습                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                Layer 1: Market Regime Classifier             │
│                      (TFT/LightGBM)                          │
│         추세장/횡보장/고변동성 등 7가지 시장 상태 분류        │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🎓 Triple Barrier Labeling
Revolutionary approach to creating training labels:
- **Profit Target**: +1.5% take profit
- **Stop Loss**: -0.5% stop loss
- **Time Limit**: 60 minutes maximum holding
- **Action Labels**: LONG, SHORT, or NEUTRAL based on future outcomes

### 🛡️ 3-Tier Risk Management

#### Level 1: Position Level
- Max position size: 8% of capital
- Max leverage: 5x
- Stop loss: 0.5%
- Take profit: 1.5%

#### Level 2: Daily Level
- Max daily loss: 2%
- Max trades: 15 per day
- Consecutive loss limit: 8

#### Level 3: System Level (Emergency Shutdown)
- Max drawdown: 5%
- Minimum Sharpe ratio: 0.5
- Black Swan detection: 3x normal volatility

### 📊 Data Strategy (Binance Futures)

#### Tier 1: Essential Data (Phase 1)
- ✅ OHLCV (1m, 5m, 15m, 1h)
- ✅ Funding Rate (real-time + historical)
- ✅ Order Book (top 10 levels)
- ✅ Recent Trades (last 1000)

#### Tier 2: Advanced Data (Phase 2)
- 🔜 Open Interest
- 🔜 Liquidation Data
- 🔜 Long/Short Ratio

### 🧪 Walk-Forward Validation
- Training window: 6 months
- Testing window: 1 month
- Step size: 1 month
- Prevents overfitting to historical data

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Binance Futures Account (Testnet for paper trading)

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd webapp

# 2. Copy environment configuration
cp .env.example .env
# Edit .env with your API keys and settings

# 3. Start services with Docker Compose
docker-compose up -d

# 4. Verify services are running
docker-compose ps

# 5. View logs
docker-compose logs -f trading_app
```

### Manual Setup (Without Docker)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup database
# Install PostgreSQL + TimescaleDB
# Run: psql -U postgres -f scripts/init_db.sql

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Run data collection
python -m src.data_collection.main

# 6. Train models
python -m src.models.train_all

# 7. Start trading system
python -m src.main
```

---

## 📁 Project Structure

```
webapp/
├── src/
│   ├── data_collection/          # Binance data collection pipeline
│   │   ├── binance_client.py    # CCXT + WebSocket integration
│   │   ├── ohlcv_collector.py   # Multi-timeframe OHLCV
│   │   ├── funding_collector.py # Funding rate collection
│   │   └── orderbook_collector.py # Order book & trades
│   │
│   ├── data_processing/          # Data normalization & feature engineering
│   │   ├── normalizer.py        # Price → Percentage conversion
│   │   ├── feature_engineer.py  # Technical indicators
│   │   └── triple_barrier.py    # Triple Barrier labeling
│   │
│   ├── models/
│   │   ├── layer1_regime/       # Market regime classification
│   │   │   ├── lightgbm_classifier.py
│   │   │   └── tft_classifier.py
│   │   │
│   │   ├── layer2_pattern/      # Pattern recognition
│   │   │   ├── tsmixer.py
│   │   │   └── patchtst.py
│   │   │
│   │   ├── layer3_signal/       # Signal generation
│   │   │   ├── tabnet_signal.py
│   │   │   ├── ft_transformer.py
│   │   │   └── catboost_signal.py
│   │   │
│   │   └── layer4_position/     # Position management
│   │       ├── sac_agent.py
│   │       ├── ppo_agent.py
│   │       └── rule_based.py
│   │
│   ├── backtesting/              # Walk-Forward backtesting engine
│   │   ├── engine.py            # Main backtesting engine
│   │   ├── walk_forward.py      # Walk-Forward validation
│   │   └── metrics.py           # Performance metrics
│   │
│   ├── risk_management/          # 3-tier risk management
│   │   ├── position_risk.py     # Position-level limits
│   │   ├── daily_risk.py        # Daily limits
│   │   └── system_risk.py       # Emergency shutdown
│   │
│   ├── trading/                  # Live trading execution
│   │   ├── executor.py          # Order execution
│   │   ├── position_tracker.py  # Position tracking
│   │   └── paper_trading.py     # Paper trading mode
│   │
│   └── utils/                    # Utilities
│       ├── config.py            # Configuration management
│       ├── logger.py            # Logging setup
│       └── database.py          # Database connections
│
├── config/                       # Configuration files
├── tests/                        # Unit & integration tests
├── notebooks/                    # Jupyter notebooks for analysis
├── scripts/                      # Utility scripts
├── data/                         # Data storage
│   ├── raw/                     # Raw market data
│   ├── processed/               # Processed features
│   └── models/                  # Trained models
├── logs/                         # Application logs
└── docs/                         # Documentation
```

---

## 🎯 Roadmap

### Phase 1: Stable Foundation (2-3 months) ✅ In Progress
**Goal**: Build a stable, loss-free baseline system

**Target Performance**:
- Sharpe Ratio: > 0.8
- Max Drawdown: < 8%
- Win Rate: > 52%

**Key Tasks**:
- [x] Binance data pipeline
- [x] Percentage-based normalization
- [x] Triple Barrier labeling
- [ ] Walk-Forward backtesting engine
- [ ] LightGBM + XGBoost signal generation
- [ ] Rule-based position management
- [ ] 3-tier risk management system

### Phase 2: Performance Optimization (3-4 months)
**Goal**: Upgrade each layer for enhanced performance

**Upgrades**:
1. Signal Generator: XGBoost → TabNet + FT-Transformer (+20-30% win rate)
2. Position Manager: Rule-based → SAC Reinforcement Learning
3. Regime Classifier: LightGBM → TFT

**Expected Performance**:
- Sharpe Ratio: > 1.2
- Max Drawdown: < 6%
- Win Rate: > 58%

### Phase 3: Production Deployment (Ongoing)
**Goal**: Live trading with continuous monitoring

**Strategy**:
- Start with 1-2% of capital
- Max 5x leverage
- Paper trading validation: 3 months minimum
- Gradual scaling based on proven performance

---

## 📊 Performance Targets

### Year 1: System Stabilization
- **Target Return**: 10-15% (annualized)
- **Primary Goal**: System stability and loss prevention
- **Sharpe Ratio**: 0.8-1.2
- **Max Drawdown**: < 8%

### Year 2: Optimization
- **Target Return**: 15-25% (annualized)
- **Primary Goal**: Performance optimization and scaling
- **Sharpe Ratio**: 1.2-1.8
- **Max Drawdown**: < 6%

### Long-term: Sustainable Excellence
- **Sustainable Return**: 20-30% (annualized)
- **Note**: This level is top 10% of professional hedge funds

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_triple_barrier.py -v

# Run integration tests
pytest tests/integration/ -v
```

---

## 📈 Monitoring & Dashboards

### TensorBoard (Model Training)
```bash
tensorboard --logdir=logs/tensorboard
# Access: http://localhost:6006
```

### Trading Dashboard (Real-time Monitoring)
```bash
python -m src.dashboard.app
# Access: http://localhost:8050
```

### API Documentation
```bash
# Start API server
python -m src.api.main
# Swagger UI: http://localhost:8000/docs
```

---

## 🔐 Security Best Practices

1. **Never commit `.env` file** - Contains API keys
2. **Use testnet for paper trading** - Validate before going live
3. **Start with small capital** - 1-2% of account
4. **Enable IP whitelisting** - On Binance API settings
5. **Monitor daily** - Check system health and performance
6. **Set up alerts** - Get notified of anomalies

---

## 🤝 Contributing

This is a production trading system. Contributions should be:
1. Well-tested with unit tests
2. Documented with clear docstrings
3. Validated with backtests
4. Reviewed for security implications

---

## ⚠️ Disclaimer

**IMPORTANT**: This is an automated trading system dealing with real money.

- **Trading cryptocurrencies involves significant risk**
- **Past performance does not guarantee future results**
- **Use at your own risk**
- **Start with paper trading and small amounts**
- **Never invest more than you can afford to lose**
- **This is not financial advice**

---

## 📚 References & Inspiration

- [Triple Barrier Method](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419) - Marcos López de Prado
- [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363) - Google Research
- [TabNet](https://arxiv.org/abs/1908.07442) - Attention-based tabular learning
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) - Berkeley RL

---

## 📞 Support & Contact

For issues, questions, or contributions:
- Open an issue on GitHub
- Check documentation in `/docs`
- Review examples in `/notebooks`

---

## 📄 License

[Specify your license here]

---

**Built with ❤️ and rigorous risk management**

*Remember: Survival first, profits second.*
