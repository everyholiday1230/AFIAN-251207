"""
Database Connections
====================

Database connection management for PostgreSQL/TimescaleDB and Redis.
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional

import redis
from sqlalchemy import create_engine, event, pool
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("database")

# Base class for SQLAlchemy models
Base = declarative_base()


class DatabaseManager:
    """Manages database connections (PostgreSQL/TimescaleDB)."""
    
    def __init__(self):
        self.engine: Optional[create_engine] = None
        self.async_engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[sessionmaker] = None
        self.async_session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
    
    def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            logger.warning("Database already initialized")
            return
        
        try:
            # Synchronous engine
            self.engine = create_engine(
                config.database.database_url,
                poolclass=pool.QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,   # Recycle connections after 1 hour
                echo=False,
            )
            
            # Async engine (for async operations)
            async_url = config.database.database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
            self.async_engine = create_async_engine(
                async_url,
                poolclass=NullPool,  # Use NullPool for async
                echo=False,
            )
            
            # Session factories
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
            )
            
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            logger.info("✅ Database initialized successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a synchronous database session.
        
        Usage:
            with db.get_session() as session:
                result = session.query(Model).all()
        """
        if not self._initialized:
            self.initialize()
        
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an asynchronous database session.
        
        Usage:
            async with db.get_async_session() as session:
                result = await session.execute(query)
        """
        if not self._initialized:
            self.initialize()
        
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error: {e}")
            raise
        finally:
            await session.close()
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
        if self.async_engine:
            asyncio.run(self.async_engine.dispose())
        
        self._initialized = False
        logger.info("Database connections closed")


class RedisManager:
    """Manages Redis connections for caching and real-time data."""
    
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self._initialized = False
    
    def initialize(self):
        """Initialize Redis connection."""
        if self._initialized:
            logger.warning("Redis already initialized")
            return
        
        try:
            self.client = redis.from_url(
                config.database.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            
            # Test connection
            self.client.ping()
            
            logger.info("✅ Redis initialized successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis: {e}")
            raise
    
    def get_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self._initialized:
            self.initialize()
        return self.client
    
    def cache_set(
        self,
        key: str,
        value: str,
        expire_seconds: Optional[int] = None
    ):
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expire_seconds: Optional expiration time in seconds
        """
        if not self._initialized:
            self.initialize()
        
        try:
            if expire_seconds:
                self.client.setex(key, expire_seconds, value)
            else:
                self.client.set(key, value)
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
    
    def cache_get(self, key: str) -> Optional[str]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            return None
    
    def cache_delete(self, key: str):
        """Delete a key from cache."""
        if not self._initialized:
            self.initialize()
        
        try:
            self.client.delete(key)
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
    
    def publish(self, channel: str, message: str):
        """
        Publish a message to a channel.
        
        Args:
            channel: Channel name
            message: Message to publish
        """
        if not self._initialized:
            self.initialize()
        
        try:
            self.client.publish(channel, message)
        except Exception as e:
            logger.error(f"Redis publish error: {e}")
    
    def subscribe(self, channels: list):
        """
        Subscribe to channels.
        
        Args:
            channels: List of channel names
        
        Returns:
            PubSub object
        """
        if not self._initialized:
            self.initialize()
        
        try:
            pubsub = self.client.pubsub()
            pubsub.subscribe(*channels)
            return pubsub
        except Exception as e:
            logger.error(f"Redis subscribe error: {e}")
            return None
    
    def close(self):
        """Close Redis connection."""
        if self.client:
            self.client.close()
        
        self._initialized = False
        logger.info("Redis connection closed")


# Global instances
db_manager = DatabaseManager()
redis_manager = RedisManager()


# Convenience functions
def get_db_session() -> Generator[Session, None, None]:
    """Get database session (convenience function)."""
    return db_manager.get_session()


async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session (convenience function)."""
    async with db_manager.get_async_session() as session:
        yield session


def get_redis() -> redis.Redis:
    """Get Redis client (convenience function)."""
    return redis_manager.get_client()


# Database utilities
def execute_raw_sql(query: str, params: Optional[dict] = None):
    """
    Execute raw SQL query.
    
    Args:
        query: SQL query string
        params: Optional query parameters
    
    Returns:
        Query result
    """
    with db_manager.get_session() as session:
        result = session.execute(query, params or {})
        return result.fetchall()


async def execute_raw_sql_async(query: str, params: Optional[dict] = None):
    """
    Execute raw SQL query asynchronously.
    
    Args:
        query: SQL query string
        params: Optional query parameters
    
    Returns:
        Query result
    """
    async with db_manager.get_async_session() as session:
        result = await session.execute(query, params or {})
        return result.fetchall()


# Health check functions
def check_database_health() -> bool:
    """Check if database is healthy."""
    try:
        with db_manager.get_session() as session:
            session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def check_redis_health() -> bool:
    """Check if Redis is healthy."""
    try:
        redis_manager.get_client().ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


def check_all_connections() -> dict:
    """
    Check health of all database connections.
    
    Returns:
        Dictionary with health status
    """
    return {
        "database": check_database_health(),
        "redis": check_redis_health(),
    }


if __name__ == "__main__":
    # Test database connections
    print("=== Database Connection Test ===\n")
    
    # Initialize
    print("1. Initializing connections...")
    try:
        db_manager.initialize()
        redis_manager.initialize()
        print("✅ Initialization successful\n")
    except Exception as e:
        print(f"❌ Initialization failed: {e}\n")
        exit(1)
    
    # Test PostgreSQL
    print("2. Testing PostgreSQL...")
    try:
        with db_manager.get_session() as session:
            result = session.execute("SELECT version()").fetchone()
            print(f"✅ PostgreSQL: {result[0][:50]}...\n")
    except Exception as e:
        print(f"❌ PostgreSQL test failed: {e}\n")
    
    # Test Redis
    print("3. Testing Redis...")
    try:
        redis_client = redis_manager.get_client()
        redis_client.set("test_key", "test_value")
        value = redis_client.get("test_key")
        redis_client.delete("test_key")
        print(f"✅ Redis: Write and read successful (value: {value})\n")
    except Exception as e:
        print(f"❌ Redis test failed: {e}\n")
    
    # Health check
    print("4. Health Check...")
    health = check_all_connections()
    for service, status in health.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {service}: {'Healthy' if status else 'Unhealthy'}")
    
    # Cleanup
    print("\n5. Closing connections...")
    db_manager.close()
    redis_manager.close()
    print("✅ All connections closed")
