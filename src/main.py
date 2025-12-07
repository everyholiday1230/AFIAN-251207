"""
Main Application Entry Point
==============================

Cryptocurrency Futures AI Trading System
"""

import sys
import time
from datetime import datetime
from pathlib import Path

from src.utils.config import config
from src.utils.database import check_all_connections, db_manager, redis_manager
from src.utils.logger import get_logger

logger = get_logger("main")


def print_banner():
    """Print application banner."""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║     Crypto Futures AI Trading System v1.0.0               ║
    ║                                                            ║
    ║     "예측이 아닌 대응" - We respond optimally               ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"    Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Mode: {config.trading.trading_mode.upper()}")
    print(f"    Symbols: {', '.join(config.trading.symbols_list)}")
    print(f"    Environment: {config.system.python_env}")
    print()


def check_system_health():
    """Check system health before starting."""
    logger.info("=" * 60)
    logger.info("System Health Check")
    logger.info("=" * 60)
    
    # Check database connections
    logger.info("Checking database connections...")
    health = check_all_connections()
    
    all_healthy = all(health.values())
    
    for service, status in health.items():
        status_icon = "✅" if status else "❌"
        logger.info(f"{status_icon} {service.capitalize()}: {'Healthy' if status else 'Unhealthy'}")
    
    if not all_healthy:
        logger.error("System health check failed. Please fix issues before starting.")
        return False
    
    logger.info("=" * 60)
    logger.info("✅ All systems operational")
    logger.info("=" * 60)
    
    return True


def initialize_services():
    """Initialize all services."""
    logger.info("Initializing services...")
    
    try:
        # Initialize database connections
        db_manager.initialize()
        redis_manager.initialize()
        
        logger.info("✅ All services initialized")
        return True
    
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        return False


def main():
    """Main application entry point."""
    # Print banner
    print_banner()
    
    # Initialize logging
    logger.info("Starting Crypto Futures AI Trading System...")
    logger.info(f"Configuration: {config}")
    
    # Initialize services
    if not initialize_services():
        logger.error("Failed to initialize services. Exiting.")
        sys.exit(1)
    
    # Health check
    if not check_system_health():
        logger.error("Health check failed. Exiting.")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("System is ready!")
    logger.info("=" * 60)
    
    try:
        # Main application loop would go here
        # For now, just keep running
        logger.info("Application running. Press Ctrl+C to stop.")
        
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("\nShutdown signal received. Stopping...")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        db_manager.close()
        redis_manager.close()
        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    main()
