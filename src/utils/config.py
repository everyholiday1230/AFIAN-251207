"""
Configuration Management
=========================

Centralized configuration management using Pydantic settings.
Loads from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    postgres_user: str = "trading_admin"
    postgres_password: str = "secure_password_change_me"
    postgres_db: str = "crypto_trading"
    database_url: str = Field(
        default="postgresql://trading_admin:secure_password_change_me@localhost:5432/crypto_trading"
    )
    redis_url: str = "redis://localhost:6379/0"


class BinanceConfig(BaseSettings):
    """Binance API configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True
    binance_testnet_api_key: str = ""
    binance_testnet_api_secret: str = ""
    
    @property
    def api_key(self) -> str:
        """Get appropriate API key based on testnet mode."""
        return self.binance_testnet_api_key if self.binance_testnet else self.binance_api_key
    
    @property
    def api_secret(self) -> str:
        """Get appropriate API secret based on testnet mode."""
        return self.binance_testnet_api_secret if self.binance_testnet else self.binance_api_secret


class TradingConfig(BaseSettings):
    """Trading configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Mode
    trading_mode: str = "paper"  # paper or live
    
    # Symbols
    trading_symbols: str = "BTCUSDT"
    primary_symbol: str = "BTCUSDT"
    
    # Capital
    initial_capital: float = 10000.0
    max_position_size: float = 0.08  # 8% of capital
    max_leverage: int = 5
    
    @field_validator("trading_symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v):
        """Parse comma-separated symbols."""
        if isinstance(v, str):
            return v
        return ",".join(v)
    
    @property
    def symbols_list(self) -> List[str]:
        """Get list of trading symbols."""
        return [s.strip() for s in self.trading_symbols.split(",")]


class RiskConfig(BaseSettings):
    """Risk management configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Position Level
    stop_loss_pct: float = 0.005  # 0.5%
    take_profit_pct: float = 0.015  # 1.5%
    
    # Daily Level
    max_daily_loss: float = 0.02  # 2%
    max_daily_trades: int = 15
    max_consecutive_losses: int = 8
    
    # System Level
    max_drawdown: float = 0.05  # 5% - EMERGENCY SHUTDOWN
    min_sharpe_ratio: float = 0.5
    emergency_volatility_multiplier: float = 3.0


class ModelConfig(BaseSettings):
    """Model configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Layer 1: Regime Classifier
    regime_model: str = "lightgbm"
    regime_lookback_periods: int = 200
    
    # Layer 2: Pattern Recognizer
    pattern_model: str = "tsmixer"
    pattern_sequence_length: int = 100
    
    # Layer 3: Signal Generator
    signal_model: str = "xgboost"
    signal_confidence_threshold: float = 0.65
    
    # Layer 4: Position Manager
    position_model: str = "rule_based"  # sac, ppo, rule_based
    rl_update_frequency: int = 1000


class TripleBarrierConfig(BaseSettings):
    """Triple Barrier labeling configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    profit_target: float = 0.015  # 1.5%
    stop_loss_target: float = 0.005  # 0.5%
    time_limit_minutes: int = 60


class DataConfig(BaseSettings):
    """Data collection configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Timeframes
    timeframes: str = "1m,5m,15m,1h"
    
    # Historical data
    historical_days: int = 730  # 2 years
    
    # Real-time collection
    enable_realtime_collection: bool = True
    orderbook_depth: int = 10
    trades_limit: int = 1000
    
    # Update frequencies (seconds)
    ohlcv_update_freq: int = 60
    funding_rate_update_freq: int = 300
    orderbook_update_freq: int = 1
    
    @property
    def timeframes_list(self) -> List[str]:
        """Get list of timeframes."""
        return [tf.strip() for tf in self.timeframes.split(",")]


class BacktestConfig(BaseSettings):
    """Backtesting configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Walk-Forward Validation
    backtest_start_date: str = "2023-01-01"
    backtest_end_date: str = "2024-12-31"
    train_window_days: int = 180
    test_window_days: int = 30
    walk_forward_step_days: int = 30
    
    # Transaction Costs
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%
    slippage_bps: float = 5  # 5 basis points


class SystemConfig(BaseSettings):
    """System configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    python_env: str = "production"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/trading.log"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"
    
    # Performance
    max_workers: int = 4
    enable_jit: bool = True
    use_gpu: bool = False
    
    # Monitoring
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_tensorboard: bool = True
    tensorboard_port: int = 6006
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 2


class Config:
    """Main configuration class that combines all sub-configs."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.binance = BinanceConfig()
        self.trading = TradingConfig()
        self.risk = RiskConfig()
        self.model = ModelConfig()
        self.triple_barrier = TripleBarrierConfig()
        self.data = DataConfig()
        self.backtest = BacktestConfig()
        self.system = SystemConfig()
        
        # Project paths
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.data_dir / "models"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  trading_mode={self.trading.trading_mode},\n"
            f"  symbols={self.trading.symbols_list},\n"
            f"  testnet={self.binance.binance_testnet}\n"
            f")"
        )


# Global config instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    print("=== Configuration Test ===\n")
    print(f"Database URL: {config.database.database_url}")
    print(f"Redis URL: {config.database.redis_url}")
    print(f"\nTrading Mode: {config.trading.trading_mode}")
    print(f"Symbols: {config.trading.symbols_list}")
    print(f"Initial Capital: ${config.trading.initial_capital:,.2f}")
    print(f"\nRisk Management:")
    print(f"  Stop Loss: {config.risk.stop_loss_pct * 100}%")
    print(f"  Take Profit: {config.risk.take_profit_pct * 100}%")
    print(f"  Max Daily Loss: {config.risk.max_daily_loss * 100}%")
    print(f"  Max Drawdown: {config.risk.max_drawdown * 100}%")
    print(f"\nTimeframes: {config.data.timeframes_list}")
    print(f"\nProject Root: {config.project_root}")
