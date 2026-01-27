"""
AirType Cache Manager

This module provides Redis caching utilities for the application.
"""

from typing import Optional, Any, Union
import json
import logging
import os

import redis
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Redis cache manager with connection pooling and automatic serialization.
    """
    
    _instance: Optional["CacheManager"] = None
    _client: Optional[redis.Redis] = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        key_prefix: str = "airtype:",
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            key_prefix: Prefix for all cache keys
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self._initialized = True
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        
        self._connect()
    
    def _connect(self):
        """Establish Redis connection."""
        try:
            if self.redis_url.startswith("memory://"):
                logger.warning("Using memory storage (Redis not configured)")
                self._client = None
                self._memory_store = {}
            else:
                self._client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                self._client.ping()
                logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using memory storage.")
            self._client = None
            self._memory_store = {}
    
    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.key_prefix}{key}"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.1))
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
        
        Returns:
            Cached value or default
        """
        full_key = self._make_key(key)
        
        try:
            if self._client:
                value = self._client.get(full_key)
                if value is None:
                    return default
                return json.loads(value)
            else:
                return self._memory_store.get(full_key, default)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return default
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.1))
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: TTL in seconds (uses default if None)
        
        Returns:
            True if successful
        """
        full_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        
        try:
            serialized = json.dumps(value)
            
            if self._client:
                self._client.setex(full_key, ttl, serialized)
            else:
                self._memory_store[full_key] = json.loads(serialized)
            
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if successful
        """
        full_key = self._make_key(key)
        
        try:
            if self._client:
                self._client.delete(full_key)
            else:
                self._memory_store.pop(full_key, None)
            
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Glob pattern
        
        Returns:
            Number of keys deleted
        """
        full_pattern = self._make_key(pattern)
        count = 0
        
        try:
            if self._client:
                for key in self._client.scan_iter(full_pattern):
                    self._client.delete(key)
                    count += 1
            else:
                import fnmatch
                keys_to_delete = [
                    k for k in self._memory_store
                    if fnmatch.fnmatch(k, full_pattern)
                ]
                for key in keys_to_delete:
                    del self._memory_store[key]
                    count += 1
        except Exception as e:
            logger.error(f"Cache delete pattern error: {e}")
        
        return count
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if key exists
        """
        full_key = self._make_key(key)
        
        try:
            if self._client:
                return bool(self._client.exists(full_key))
            else:
                return full_key in self._memory_store
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    def get_or_set(
        self,
        key: str,
        factory: callable,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get value from cache or compute and store it.
        
        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: TTL in seconds
        
        Returns:
            Cached or computed value
        """
        value = self.get(key)
        
        if value is not None:
            return value
        
        value = factory()
        self.set(key, value, ttl)
        
        return value
    
    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter.
        
        Args:
            key: Cache key
            amount: Amount to increment by
        
        Returns:
            New value
        """
        full_key = self._make_key(key)
        
        try:
            if self._client:
                return self._client.incrby(full_key, amount)
            else:
                current = self._memory_store.get(full_key, 0)
                new_value = current + amount
                self._memory_store[full_key] = new_value
                return new_value
        except Exception as e:
            logger.error(f"Cache increment error: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        stats = {
            "connected": self._client is not None,
            "prefix": self.key_prefix,
            "default_ttl": self.default_ttl,
        }
        
        try:
            if self._client:
                info = self._client.info("memory")
                stats["used_memory"] = info.get("used_memory_human", "unknown")
                stats["connected_clients"] = self._client.info("clients").get("connected_clients", 0)
            else:
                stats["memory_store_size"] = len(self._memory_store)
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    def flush(self):
        """Flush all keys with this prefix."""
        self.delete_pattern("*")


def cached(
    key_pattern: str,
    ttl: Optional[int] = None,
    cache_manager: Optional[CacheManager] = None,
):
    """
    Decorator for caching function results.
    
    Args:
        key_pattern: Cache key pattern (can include {arg_name} placeholders)
        ttl: TTL in seconds
        cache_manager: CacheManager instance
    
    Returns:
        Decorator function
    """
    def decorator(f):
        from functools import wraps
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            cm = cache_manager or CacheManager()
            
            # Build cache key from pattern
            key = key_pattern
            
            # Replace placeholders with argument values
            import inspect
            sig = inspect.signature(f)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for param_name, param_value in bound.arguments.items():
                key = key.replace(f"{{{param_name}}}", str(param_value))
            
            # Try to get from cache
            cached_value = cm.get(key)
            if cached_value is not None:
                return cached_value
            
            # Compute and cache
            result = f(*args, **kwargs)
            cm.set(key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator
