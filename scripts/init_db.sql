-- ===================================================================
-- Crypto Futures AI Trading System - Database Initialization
-- ===================================================================
-- TimescaleDB-optimized schema for high-frequency trading data
-- ===================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ===================================================================
-- Table 1: OHLCV Data (Multi-timeframe)
-- ===================================================================
CREATE TABLE IF NOT EXISTS ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,  -- 1m, 5m, 15m, 1h, 4h, 1d
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(20, 8) NOT NULL,
    quote_volume NUMERIC(20, 8),
    trades INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe 
    ON ohlcv (symbol, timeframe, time DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_time 
    ON ohlcv (time DESC);

-- Add compression policy (compress data older than 7 days)
ALTER TABLE ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe'
);

SELECT add_compression_policy('ohlcv', INTERVAL '7 days', if_not_exists => TRUE);

-- ===================================================================
-- Table 2: Funding Rate (Futures-specific)
-- ===================================================================
CREATE TABLE IF NOT EXISTS funding_rate (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    funding_rate NUMERIC(12, 8) NOT NULL,
    mark_price NUMERIC(20, 8),
    index_price NUMERIC(20, 8),
    estimated_settle_price NUMERIC(20, 8),
    next_funding_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('funding_rate', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_funding_rate_symbol 
    ON funding_rate (symbol, time DESC);

-- ===================================================================
-- Table 3: Order Book Snapshots
-- ===================================================================
CREATE TABLE IF NOT EXISTS orderbook_snapshot (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    bids JSONB NOT NULL,  -- Array of [price, quantity]
    asks JSONB NOT NULL,
    bid_volume NUMERIC(20, 8),
    ask_volume NUMERIC(20, 8),
    spread NUMERIC(20, 8),
    imbalance NUMERIC(8, 4),  -- (bid_vol - ask_vol) / (bid_vol + ask_vol)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('orderbook_snapshot', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_orderbook_symbol 
    ON orderbook_snapshot (symbol, time DESC);

-- Compression
ALTER TABLE orderbook_snapshot SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('orderbook_snapshot', INTERVAL '1 day', if_not_exists => TRUE);

-- ===================================================================
-- Table 4: Recent Trades (Tick Data)
-- ===================================================================
CREATE TABLE IF NOT EXISTS trades (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    trade_id BIGINT NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    quote_quantity NUMERIC(20, 8),
    is_buyer_maker BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_trades_symbol 
    ON trades (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_trade_id 
    ON trades (trade_id);

-- Compression
ALTER TABLE trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('trades', INTERVAL '3 days', if_not_exists => TRUE);

-- ===================================================================
-- Table 5: Processed Features (ML-ready data)
-- ===================================================================
CREATE TABLE IF NOT EXISTS processed_features (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Normalized price changes (percentage)
    price_change_1 NUMERIC(10, 6),
    price_change_5 NUMERIC(10, 6),
    price_change_15 NUMERIC(10, 6),
    price_change_60 NUMERIC(10, 6),
    
    -- Volume changes
    volume_change NUMERIC(10, 6),
    volume_ma_ratio NUMERIC(10, 6),
    
    -- Technical indicators (normalized)
    rsi_14 NUMERIC(8, 4),
    macd NUMERIC(10, 6),
    macd_signal NUMERIC(10, 6),
    bb_position NUMERIC(8, 4),  -- Position within Bollinger Bands
    
    -- Market microstructure
    orderbook_imbalance NUMERIC(8, 4),
    trade_intensity NUMERIC(10, 6),
    buy_sell_ratio NUMERIC(8, 4),
    
    -- Futures-specific
    funding_rate NUMERIC(12, 8),
    funding_rate_ma NUMERIC(12, 8),
    
    -- Volatility
    volatility_1h NUMERIC(10, 6),
    volatility_24h NUMERIC(10, 6),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('processed_features', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_processed_features_symbol 
    ON processed_features (symbol, timeframe, time DESC);

-- ===================================================================
-- Table 6: Triple Barrier Labels
-- ===================================================================
CREATE TABLE IF NOT EXISTS triple_barrier_labels (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    entry_price NUMERIC(20, 8) NOT NULL,
    
    -- Barrier outcomes
    label VARCHAR(10) NOT NULL,  -- LONG, SHORT, NEUTRAL
    exit_price NUMERIC(20, 8),
    exit_time TIMESTAMPTZ,
    return_pct NUMERIC(10, 6),
    holding_minutes INTEGER,
    
    -- Barrier details
    profit_target NUMERIC(10, 6),
    stop_loss NUMERIC(10, 6),
    time_limit INTEGER,
    
    -- Exit reason
    exit_reason VARCHAR(20),  -- PROFIT, LOSS, TIMEOUT
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('triple_barrier_labels', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_labels_symbol 
    ON triple_barrier_labels (symbol, timeframe, time DESC);
CREATE INDEX IF NOT EXISTS idx_labels_label 
    ON triple_barrier_labels (label);

-- ===================================================================
-- Table 7: Model Predictions
-- ===================================================================
CREATE TABLE IF NOT EXISTS model_predictions (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    
    -- Predictions
    prediction VARCHAR(10) NOT NULL,  -- LONG, SHORT, NEUTRAL
    confidence NUMERIC(6, 4),
    
    -- Probabilities
    prob_long NUMERIC(6, 4),
    prob_short NUMERIC(6, 4),
    prob_neutral NUMERIC(6, 4),
    
    -- Market regime
    market_regime VARCHAR(20),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('model_predictions', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol_model 
    ON model_predictions (symbol, model_name, time DESC);

-- ===================================================================
-- Table 8: Trades (System execution)
-- ===================================================================
CREATE TABLE IF NOT EXISTS system_trades (
    id SERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    
    -- Order details
    order_id VARCHAR(50) UNIQUE,
    side VARCHAR(10) NOT NULL,  -- LONG, SHORT
    entry_price NUMERIC(20, 8) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    leverage INTEGER,
    
    -- Exit details
    exit_price NUMERIC(20, 8),
    exit_time TIMESTAMPTZ,
    realized_pnl NUMERIC(20, 8),
    realized_pnl_pct NUMERIC(10, 6),
    
    -- Fees
    entry_fee NUMERIC(20, 8),
    exit_fee NUMERIC(20, 8),
    funding_fees NUMERIC(20, 8),
    
    -- Risk management
    stop_loss NUMERIC(20, 8),
    take_profit NUMERIC(20, 8),
    
    -- Model info
    model_signal VARCHAR(50),
    signal_confidence NUMERIC(6, 4),
    
    -- Status
    status VARCHAR(20),  -- OPEN, CLOSED, CANCELLED
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_trades_symbol 
    ON system_trades (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_system_trades_status 
    ON system_trades (status);
CREATE INDEX IF NOT EXISTS idx_system_trades_order_id 
    ON system_trades (order_id);

-- ===================================================================
-- Table 9: Portfolio State (Time-series of account state)
-- ===================================================================
CREATE TABLE IF NOT EXISTS portfolio_state (
    time TIMESTAMPTZ NOT NULL,
    
    -- Account balance
    total_equity NUMERIC(20, 8) NOT NULL,
    available_balance NUMERIC(20, 8),
    used_margin NUMERIC(20, 8),
    
    -- Positions
    open_positions INTEGER,
    total_position_value NUMERIC(20, 8),
    
    -- Performance
    unrealized_pnl NUMERIC(20, 8),
    realized_pnl_today NUMERIC(20, 8),
    total_realized_pnl NUMERIC(20, 8),
    
    -- Risk metrics
    current_drawdown NUMERIC(10, 6),
    max_drawdown NUMERIC(10, 6),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('portfolio_state', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_portfolio_state_time 
    ON portfolio_state (time DESC);

-- ===================================================================
-- Table 10: Risk Events (Alert system)
-- ===================================================================
CREATE TABLE IF NOT EXISTS risk_events (
    id SERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    
    event_type VARCHAR(50) NOT NULL,  -- DAILY_LOSS_LIMIT, MAX_DRAWDOWN, etc.
    severity VARCHAR(20) NOT NULL,    -- WARNING, CRITICAL, EMERGENCY
    
    details JSONB,
    action_taken VARCHAR(100),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_events_time 
    ON risk_events (time DESC);
CREATE INDEX IF NOT EXISTS idx_risk_events_severity 
    ON risk_events (severity);

-- ===================================================================
-- Table 11: Model Performance Tracking
-- ===================================================================
CREATE TABLE IF NOT EXISTS model_performance (
    time TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    
    -- Performance metrics
    accuracy NUMERIC(6, 4),
    precision NUMERIC(6, 4),
    recall NUMERIC(6, 4),
    f1_score NUMERIC(6, 4),
    
    -- Trading metrics
    win_rate NUMERIC(6, 4),
    avg_profit NUMERIC(10, 6),
    avg_loss NUMERIC(10, 6),
    profit_factor NUMERIC(10, 4),
    sharpe_ratio NUMERIC(10, 4),
    
    -- Sample size
    total_predictions INTEGER,
    total_trades INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('model_performance', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_model_performance 
    ON model_performance (model_name, time DESC);

-- ===================================================================
-- Continuous Aggregates (Pre-computed views for fast queries)
-- ===================================================================

-- 1-hour OHLCV aggregation from 1-minute data
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS time,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    sum(trades) AS trades
FROM ohlcv
WHERE timeframe = '1m'
GROUP BY time_bucket('1 hour', time), symbol;

-- Add refresh policy
SELECT add_continuous_aggregate_policy('ohlcv_1h',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Daily portfolio performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_performance
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS date,
    last(total_equity, time) AS ending_equity,
    first(total_equity, time) AS starting_equity,
    last(total_equity, time) - first(total_equity, time) AS daily_pnl,
    ((last(total_equity, time) - first(total_equity, time)) / first(total_equity, time) * 100) AS daily_return_pct,
    max(max_drawdown) AS max_drawdown
FROM portfolio_state
GROUP BY time_bucket('1 day', time);

SELECT add_continuous_aggregate_policy('daily_performance',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ===================================================================
-- Data Retention Policies
-- ===================================================================

-- Keep raw 1-minute OHLCV for 90 days
SELECT add_retention_policy('ohlcv', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep orderbook snapshots for 7 days (very high frequency)
SELECT add_retention_policy('orderbook_snapshot', INTERVAL '7 days', if_not_exists => TRUE);

-- Keep trades for 30 days
SELECT add_retention_policy('trades', INTERVAL '30 days', if_not_exists => TRUE);

-- Keep predictions for 180 days
SELECT add_retention_policy('model_predictions', INTERVAL '180 days', if_not_exists => TRUE);

-- Keep processed features for 365 days
SELECT add_retention_policy('processed_features', INTERVAL '365 days', if_not_exists => TRUE);

-- Keep labels indefinitely (small table, valuable for retraining)
-- Keep system trades indefinitely (critical for analysis)
-- Keep portfolio state indefinitely (needed for performance tracking)

-- ===================================================================
-- Functions and Triggers
-- ===================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for system_trades
CREATE TRIGGER update_system_trades_updated_at
    BEFORE UPDATE ON system_trades
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ===================================================================
-- Initial Data Setup
-- ===================================================================

-- Create admin user (optional)
-- For application-level user management if needed

-- ===================================================================
-- Grants (Adjust according to your security requirements)
-- ===================================================================

-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO trading_app;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO trading_app;

-- ===================================================================
-- Database Initialization Complete
-- ===================================================================

-- Verify installation
SELECT * FROM timescaledb_information.hypertables;
