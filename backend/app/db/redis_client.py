"""
Redis client and utility functions for the YouTube Summarizer API.

This module provides optimized Redis connection handling and caching utilities
designed for resource-constrained environments.
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, Union, List
import redis.asyncio as redis
from datetime import datetime, timezone

from ..core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Global Redis client
redis_client: Optional[redis.Redis] = None

async def connect_to_redis() -> redis.Redis:
    """
    Connect to Redis with optimized settings for resource-constrained environments.
    
    Returns:
        redis.Redis: Redis client
    """
    global redis_client
    
    settings = get_settings()
    cache_settings = settings.cache_settings
    
    # If already connected, return existing client
    if redis_client is not None:
        return redis_client
    
    try:
        # Configure Redis client with optimized settings
        redis_client = redis.from_url(
            cache_settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=settings.timeout_settings.REDIS_OPERATION_TIMEOUT,
            socket_connect_timeout=settings.timeout_settings.REDIS_OPERATION_TIMEOUT,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True,
            max_connections=cache_settings.REDIS_MAX_CONNECTIONS
        )
        
        # Test connection
        await redis_client.ping()
        logger.info(f"Connected to Redis at {cache_settings.REDIS_URL}")
        
        # Get Redis info
        info = await redis_client.info()
        redis_version = info.get("redis_version", "unknown")
        used_memory = info.get("used_memory_human", "unknown")
        max_memory = info.get("maxmemory_human", "unknown")
        max_memory_policy = info.get("maxmemory_policy", "unknown")
        
        logger.info(f"Redis version: {redis_version}")
        logger.info(f"Redis memory usage: {used_memory} / {max_memory}")
        logger.info(f"Redis maxmemory policy: {max_memory_policy}")
        
        # Check if maxmemory policy is set to allkeys-lru
        if max_memory_policy != "allkeys-lru":
            logger.warning(
                f"Redis maxmemory policy is set to '{max_memory_policy}', "
                f"but 'allkeys-lru' is recommended for optimal caching"
            )
        
        return redis_client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise

async def close_redis_connection() -> None:
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
        logger.info("Redis connection closed")

async def get_redis() -> redis.Redis:
    """
    Get Redis client instance.
    
    Returns:
        redis.Redis: Redis client
    """
    global redis_client
    if redis_client is None:
        await connect_to_redis()
    return redis_client

async def set_cache(
    key: str,
    value: Any,
    expiry_seconds: Optional[int] = None,
    nx: bool = False
) -> bool:
    """
    Set a value in the cache with optional expiration.
    
    Args:
        key: Cache key
        value: Value to cache (will be JSON serialized)
        expiry_seconds: Optional expiration time in seconds
        nx: If True, only set the key if it does not already exist
    
    Returns:
        bool: True if successful, False otherwise
    """
    global redis_client
    if redis_client is None:
        await connect_to_redis()
    
    try:
        # Add timestamp to value for LRU implementation
        if isinstance(value, dict):
            value['_cached_at'] = datetime.now(timezone.utc).isoformat()
            value['_cache_key'] = key
        
        # Serialize value to JSON
        serialized_value = json.dumps(value)
        
        # Calculate size for logging
        size_kb = len(serialized_value) / 1024
        
        # Set with or without expiration
        if expiry_seconds:
            result = await redis_client.setex(
                name=key,
                time=expiry_seconds,
                value=serialized_value,
                nx=nx
            )
            logger.debug(f"Cached data with key: {key} (expires in {expiry_seconds}s, size: {size_kb:.2f}KB)")
        else:
            result = await redis_client.set(
                name=key,
                value=serialized_value,
                nx=nx
            )
            logger.debug(f"Cached data with key: {key} (permanent storage, size: {size_kb:.2f}KB)")
        
        return bool(result)
    except Exception as e:
        logger.error(f"Error setting cache for key {key}: {e}")
        return False

async def get_cache(key: str) -> Optional[Any]:
    """
    Get a value from the cache.
    
    Args:
        key: Cache key
    
    Returns:
        The cached value or None if not found
    """
    global redis_client
    if redis_client is None:
        await connect_to_redis()
    
    try:
        # Get from cache
        cached_value = await redis_client.get(key)
        if cached_value:
            # Deserialize from JSON
            value = json.loads(cached_value)
            logger.debug(f"Cache hit for key: {key}")
            return value
        logger.debug(f"Cache miss for key: {key}")
        return None
    except Exception as e:
        logger.error(f"Error getting cache for key {key}: {e}")
        return None

async def delete_cache(key: str) -> bool:
    """
    Delete a value from the cache.
    
    Args:
        key: Cache key
    
    Returns:
        bool: True if successful, False otherwise
    """
    global redis_client
    if redis_client is None:
        await connect_to_redis()
    
    try:
        result = await redis_client.delete(key)
        logger.debug(f"Deleted cache for key: {key}")
        return bool(result)
    except Exception as e:
        logger.error(f"Error deleting cache for key {key}: {e}")
        return False

async def clear_cache() -> bool:
    """
    Clear all cache.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global redis_client
    if redis_client is None:
        await connect_to_redis()
    
    try:
        await redis_client.flushdb()
        logger.info("Cache cleared")
        return True
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return False

async def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    global redis_client
    if redis_client is None:
        await connect_to_redis()
    
    try:
        # Get memory info
        info = await redis_client.info("memory")
        stats = {
            "status": "Connected",
            "used_memory_human": info.get("used_memory_human", "Unknown"),
            "maxmemory_human": info.get("maxmemory_human", "Unknown"),
            "memory_percent": "Unknown",
            "total_keys": await redis_client.dbsize(),
            "transcript_keys": len(await redis_client.keys("transcript:*")),
            "video_info_keys": len(await redis_client.keys("video_info:*")),
            "languages_keys": len(await redis_client.keys("languages:*")),
            "summary_keys": len(await redis_client.keys("summary:*")),
            "task_keys": len(await redis_client.keys("task:*")),
            "last_cleanup": await get_cache("last_cleanup_time") or "Never",
        }
        
        # Calculate memory percentage if possible
        used_memory = int(info.get("used_memory", 0))
        max_memory = int(info.get("maxmemory", 0))
        if max_memory > 0:
            memory_percent = (used_memory / max_memory) * 100
            stats["memory_percent"] = f"{memory_percent:.2f}%"
        
        return stats
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {"status": f"Error: {str(e)}"}

async def set_with_ttl(
    key: str,
    value: Any,
    resource_type: str
) -> bool:
    """
    Set a value in the cache with TTL based on resource type.
    
    Args:
        key: Cache key
        value: Value to cache
        resource_type: Type of resource (video_metadata, transcript, summary)
    
    Returns:
        bool: True if successful, False otherwise
    """
    settings = get_settings()
    cache_settings = settings.cache_settings
    
    # Determine TTL based on resource type
    ttl = None
    if resource_type == "video_metadata":
        ttl = cache_settings.VIDEO_METADATA_TTL
    elif resource_type == "transcript":
        ttl = cache_settings.TRANSCRIPT_TTL
    elif resource_type == "summary":
        ttl = cache_settings.SUMMARY_TTL
    
    return await set_cache(key, value, expiry_seconds=ttl)

async def get_keys_by_pattern(pattern: str) -> List[str]:
    """
    Get keys matching a pattern.
    
    Args:
        pattern: Key pattern (e.g., "transcript:*")
    
    Returns:
        List of matching keys
    """
    global redis_client
    if redis_client is None:
        await connect_to_redis()
    
    try:
        return await redis_client.keys(pattern)
    except Exception as e:
        logger.error(f"Error getting keys by pattern {pattern}: {e}")
        return []

async def get_memory_usage() -> Dict[str, Any]:
    """
    Get Redis memory usage information.
    
    Returns:
        Dictionary with memory usage information
    """
    global redis_client
    if redis_client is None:
        await connect_to_redis()
    
    try:
        info = await redis_client.info("memory")
        return {
            "used_memory": int(info.get("used_memory", 0)),
            "used_memory_human": info.get("used_memory_human", "Unknown"),
            "maxmemory": int(info.get("maxmemory", 0)),
            "maxmemory_human": info.get("maxmemory_human", "Unknown"),
            "maxmemory_policy": info.get("maxmemory_policy", "Unknown"),
            "mem_fragmentation_ratio": float(info.get("mem_fragmentation_ratio", 0)),
        }
    except Exception as e:
        logger.error(f"Error getting Redis memory usage: {e}")
        return {}
