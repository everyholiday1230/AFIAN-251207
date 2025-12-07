# 🚀 Quick Start Guide

Get your crypto trading system running in 10 minutes!

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (recommended)
- Binance Futures Account (testnet for paper trading)

## Step 1: Clone & Setup

```bash
# Navigate to project directory
cd webapp

# Create environment file
cp .env.example .env

# Edit .env with your settings
nano .env  # or vim .env
```

## Step 2: Configure Your API Keys

Edit `.env` file:

```bash
# Binance Testnet (for paper trading)
BINANCE_TESTNET=true
BINANCE_TESTNET_API_KEY=your_testnet_api_key
BINANCE_TESTNET_API_SECRET=your_testnet_api_secret

# Get testnet keys from: https://testnet.binancefuture.com
```

## Step 3: Start with Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f trading_app

# Verify services are running
docker-compose ps
```

## Step 4: Manual Setup (Without Docker)

### 4.1 Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4.2 Setup Database

```bash
# Install PostgreSQL + TimescaleDB
# On Ubuntu/Debian:
sudo apt-get install postgresql-14 timescaledb-postgresql-14

# Initialize database
sudo -u postgres psql -f scripts/init_db.sql

# Or use Docker for just the database:
docker-compose up -d timescaledb redis
```

### 4.3 Run Application

```bash
# Start main application
python -m src.main
```

## Step 5: Collect Historical Data

```bash
# Run data collection script
python -m src.data_collection.collect_historical

# This will:
# - Fetch 2 years of OHLCV data
# - Collect funding rates
# - Store in TimescaleDB
```

## Step 6: Train Models

```bash
# Create features and labels
python -m scripts.prepare_training_data

# Train models
python -m scripts.train_models

# Models will be saved to data/models/
```

## Step 7: Run Backtest

```bash
# Run Walk-Forward backtest
python -m scripts.run_backtest --start 2023-01-01 --end 2024-12-31

# View results
python -m scripts.analyze_backtest_results
```

## Step 8: Paper Trading

```bash
# Start paper trading (testnet)
python -m src.trading.paper_trading

# Monitor dashboard (in browser)
# Visit: http://localhost:8050
```

## Step 9: Monitor Performance

```bash
# TensorBoard (model training)
tensorboard --logdir=logs/tensorboard
# Visit: http://localhost:6006

# Trading Dashboard (real-time)
python -m src.dashboard.app
# Visit: http://localhost:8050

# API Documentation
# Visit: http://localhost:8000/docs
```

## Quick Commands Cheat Sheet

```bash
# Start everything
docker-compose up -d

# Stop everything
docker-compose down

# View logs
docker-compose logs -f [service_name]

# Restart a service
docker-compose restart [service_name]

# Run tests
pytest tests/ -v

# Code formatting
black src/
isort src/

# Check code quality
flake8 src/
mypy src/
```

## Troubleshooting

### Database Connection Error

```bash
# Check if PostgreSQL is running
docker-compose ps timescaledb

# Verify connection
psql -h localhost -U trading_admin -d crypto_trading
```

### Binance API Error

```bash
# Test API connection
python -c "from src.data_collection.binance_client import BinanceClient; client = BinanceClient(testnet=True); print(client.fetch_ohlcv('BTC/USDT', '1h', limit=5))"
```

### Module Not Found Error

```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## Configuration Tips

### For Better Performance

```bash
# In .env, enable JIT compilation
ENABLE_JIT=true

# Use more workers (adjust based on CPU cores)
MAX_WORKERS=8

# If you have GPU
USE_GPU=true
```

### For Development

```bash
# Enable debug mode
LOG_LEVEL=DEBUG
ENABLE_DEBUG_MODE=true

# Save debug data
SAVE_DEBUG_DATA=true
```

### For Production

```bash
# Set production mode
PYTHON_ENV=production
TRADING_MODE=live  # CAUTION: Real money!

# Use production Binance API keys
BINANCE_TESTNET=false
BINANCE_API_KEY=your_production_api_key
BINANCE_API_SECRET=your_production_api_secret
```

## Safety Checklist Before Going Live

- [ ] Tested in paper trading for at least 3 months
- [ ] Sharpe ratio > 0.8
- [ ] Max drawdown < 8%
- [ ] Win rate > 52%
- [ ] All risk limits configured correctly
- [ ] Start with 1-2% of capital only
- [ ] Monitor daily for first 2 weeks
- [ ] Set up alerts (Telegram/Discord/Email)

## Next Steps

1. **Read the full documentation** in `/docs`
2. **Understand the risk management** - survival first!
3. **Test extensively** in paper trading
4. **Start small** - scale gradually
5. **Monitor continuously** - check daily

## Getting Help

- Check `README.md` for detailed architecture
- Review code comments for implementation details
- Run tests to understand component behavior
- Check logs in `logs/` directory

## Important Notes

⚠️ **Risk Warning**: Cryptocurrency trading involves significant risk. This system is provided as-is with no guarantees of profitability. Always start with paper trading and only risk capital you can afford to lose.

🎯 **Performance Expectations**: Year 1 target is 10-15% annual return with <8% drawdown. This is realistic and sustainable. Don't expect 100% returns - those who do usually lose everything.

🛡️ **Risk Management**: The system will automatically shut down if:
- Daily loss exceeds 2%
- Total drawdown exceeds 5%
- Volatility spikes 3x normal
- Consecutive losses reach 8

Never disable these safety features!

---

**Ready to trade? Start with paper trading and good luck! 🚀**
