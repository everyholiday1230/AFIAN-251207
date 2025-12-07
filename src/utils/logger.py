"""
Logging Configuration
=====================

Centralized logging using loguru with rotation and retention.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.utils.config import config


class Logger:
    """Custom logger wrapper for the trading system."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            name: Optional name for the logger module
        """
        self.name = name or "trading"
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with file rotation and formatting."""
        # Remove default handler
        logger.remove()
        
        # Console handler with colors
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=config.system.log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
        
        # File handler with rotation
        log_file = config.logs_dir / config.system.log_file.split("/")[-1]
        logger.add(
            log_file,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            level="DEBUG",  # Save all logs to file
            rotation=config.system.log_rotation,
            retention=config.system.log_retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )
        
        # Separate file for errors
        error_log_file = config.logs_dir / "error.log"
        logger.add(
            error_log_file,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}\n"
                "{exception}"
            ),
            level="ERROR",
            rotation="50 MB",
            retention="90 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )
        
        # Trading-specific log (trades, signals, positions)
        trading_log_file = config.logs_dir / "trading.log"
        logger.add(
            trading_log_file,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{message}"
            ),
            level="INFO",
            rotation="20 MB",
            retention="180 days",
            compression="zip",
            filter=lambda record: "TRADE" in record["message"] or "SIGNAL" in record["message"],
        )
    
    def get_logger(self):
        """Get logger instance."""
        return logger.bind(name=self.name)


# Global logger instance
trading_logger = Logger("trading").get_logger()


# Convenience functions
def get_logger(name: str):
    """Get a logger with specific name."""
    return Logger(name).get_logger()


def log_trade(
    action: str,
    symbol: str,
    price: float,
    quantity: float,
    side: str,
    **kwargs
):
    """
    Log trade execution.
    
    Args:
        action: Trade action (ENTRY, EXIT, etc.)
        symbol: Trading symbol
        price: Execution price
        quantity: Trade quantity
        side: LONG or SHORT
        **kwargs: Additional trade details
    """
    extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    message = (
        f"TRADE | {action} | {symbol} | {side} | "
        f"Price: {price:.2f} | Qty: {quantity:.4f}"
    )
    if extra_info:
        message += f" | {extra_info}"
    
    trading_logger.info(message)


def log_signal(
    symbol: str,
    signal: str,
    confidence: float,
    model: str,
    **kwargs
):
    """
    Log trading signal.
    
    Args:
        symbol: Trading symbol
        signal: Signal type (LONG, SHORT, NEUTRAL)
        confidence: Signal confidence (0-1)
        model: Model name that generated the signal
        **kwargs: Additional signal details
    """
    extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    message = (
        f"SIGNAL | {symbol} | {signal} | "
        f"Confidence: {confidence:.2%} | Model: {model}"
    )
    if extra_info:
        message += f" | {extra_info}"
    
    trading_logger.info(message)


def log_risk_event(
    event_type: str,
    severity: str,
    details: str,
    action: Optional[str] = None
):
    """
    Log risk management event.
    
    Args:
        event_type: Type of risk event
        severity: WARNING, CRITICAL, or EMERGENCY
        details: Event details
        action: Action taken in response
    """
    message = f"RISK | {severity} | {event_type} | {details}"
    if action:
        message += f" | Action: {action}"
    
    if severity == "EMERGENCY":
        trading_logger.critical(message)
    elif severity == "CRITICAL":
        trading_logger.error(message)
    else:
        trading_logger.warning(message)


def log_performance(
    period: str,
    metrics: dict
):
    """
    Log performance metrics.
    
    Args:
        period: Time period (DAILY, WEEKLY, etc.)
        metrics: Dictionary of performance metrics
    """
    metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
    message = f"PERFORMANCE | {period} | {metrics_str}"
    trading_logger.info(message)


# Context manager for timing operations
from contextlib import contextmanager
from time import time


@contextmanager
def log_execution_time(operation: str, logger_instance=None):
    """
    Context manager to log execution time of operations.
    
    Usage:
        with log_execution_time("Data Collection"):
            collect_data()
    """
    log = logger_instance or trading_logger
    start_time = time()
    log.debug(f"Starting: {operation}")
    
    try:
        yield
    finally:
        elapsed = time() - start_time
        log.debug(f"Completed: {operation} in {elapsed:.2f}s")


if __name__ == "__main__":
    # Test logging
    test_logger = get_logger("test")
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    
    # Test trade logging
    log_trade(
        action="ENTRY",
        symbol="BTCUSDT",
        price=45000.0,
        quantity=0.1,
        side="LONG",
        leverage=5,
        stop_loss=44775.0,
        take_profit=45675.0
    )
    
    # Test signal logging
    log_signal(
        symbol="BTCUSDT",
        signal="LONG",
        confidence=0.85,
        model="xgboost_v1",
        regime="TRENDING_UP",
        volatility=0.025
    )
    
    # Test risk event logging
    log_risk_event(
        event_type="DAILY_LOSS_LIMIT",
        severity="WARNING",
        details="Daily loss reached 1.5% (limit: 2.0%)",
        action="Reduced position sizes"
    )
    
    # Test execution time logging
    with log_execution_time("Test Operation"):
        import time
        time.sleep(0.5)
    
    print("\n✅ Logging test completed. Check logs/ directory.")
