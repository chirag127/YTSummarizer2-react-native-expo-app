"""
Cache service for the YouTube Summarizer API.

This module provides a multi-level caching implementation with:
1. In-memory LRU cache (10MB maximum)
2. Redis cache (128MB maximum)

It follows an optimized caching strategy for resource-constrained environments.
"""

import json
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from datetime import datetime, timezone
from functools import lru_cache
import sys

from ..core.config import get_settings
from ..db import redis_client

# Configure logging
logger = logging.getLogger(__name__)

# In-memory LRU cache size calculation
# Estimate bytes per cache entry and set maxsize accordingly
# Target: 10MB maximum in-memory cache
BYTES_PER_CACHE_ENTRY = 10 * 1024  # Assume 10KB per entry on average
IN_MEMORY_CACHE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
IN_MEMORY_CACHE_MAXSIZE = IN_MEMORY_CACHE_SIZE_BYTES // BYTES_PER_CACHE_ENTRY

# Cache key patterns
VIDEO_INFO_KEY_PATTERN = "video_info:{video_id}"
TRANSCRIPT_KEY_PATTERN = "transcript:{video_id}"
SUMMARY_KEY_PATTERN = "summary:{video_id}:{summary_type}:{summary_length}"
LANGUAGES_KEY_PATTERN = "languages:{video_id}"

# TTL values in seconds (will be overridden by config)
VIDEO_METADATA_TTL = 86400  # 24 hours
TRANSCRIPT_TTL = 172800  # 48 hours
SUMMARY_TTL = 259200  # 72 hours

class CacheEntry:
    """
    Class representing a cache entry with metadata.
    
    Attributes:
        key: Cache key
        value: Cached value
        created_at: Creation timestamp
        last_accessed: Last access timestamp
        ttl: Time-to-live in seconds (None for no expiration)
        size_bytes: Estimated size in bytes
    """
    
    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        size_bytes: Optional[int] = None
    ):
        """
        Initialize a cache entry.
        
        Args:
            key: Cache key
            value: Cached value
            ttl: Time-to-live in seconds (None for no expiration)
            size_bytes: Estimated size in bytes
        """
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.ttl = ttl
        
        # Calculate size if not provided
        if size_bytes is None:
            # Estimate size based on JSON serialization
            try:
                serialized = json.dumps(value)
                self.size_bytes = len(serialized.encode('utf-8'))
            except (TypeError, OverflowError):
                # Fallback for non-serializable objects
                self.size_bytes = sys.getsizeof(str(value))
        else:
            self.size_bytes = size_bytes
    
    def is_expired(self) -> bool:
        """
        Check if the cache entry is expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl
    
    def access(self) -> None:
        """Update the last accessed timestamp."""
        self.last_accessed = time.time()
    
    def get_age(self) -> float:
        """
        Get the age of the cache entry in seconds.
        
        Returns:
            Age in seconds
        """
        return time.time() - self.created_at
    
    def get_idle_time(self) -> float:
        """
        Get the idle time of the cache entry in seconds.
        
        Returns:
            Idle time in seconds
        """
        return time.time() - self.last_accessed
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the cache entry to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "key": self.key,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "ttl": self.ttl,
            "size_bytes": self.size_bytes,
            "is_expired": self.is_expired(),
            "age": self.get_age(),
            "idle_time": self.get_idle_time()
        }

class InMemoryLRUCache:
    """
    In-memory LRU cache implementation with size limit.
    
    This cache uses a dictionary for O(1) lookups and a doubly-linked list
    for O(1) insertions and removals, implementing the LRU eviction policy.
    """
    
    def __init__(self, max_size_bytes: int = IN_MEMORY_CACHE_SIZE_BYTES):
        """
        Initialize the in-memory LRU cache.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
        """
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # List of keys in LRU order
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if entry.is_expired():
            self._remove(key)
            return None
        
        # Update access order
        self._move_to_front(key)
        
        # Update last accessed timestamp
        entry.access()
        
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for no expiration)
        
        Returns:
            True if successful, False otherwise
        """
        # Create new cache entry
        entry = CacheEntry(key, value, ttl)
        
        # Check if key already exists
        if key in self.cache:
            old_entry = self.cache[key]
            self.current_size_bytes -= old_entry.size_bytes
            self._remove_from_access_order(key)
        
        # Check if new entry would exceed size limit
        if entry.size_bytes > self.max_size_bytes:
            logger.warning(
                f"Cache entry too large: {entry.size_bytes} bytes "
                f"(max: {self.max_size_bytes} bytes)"
            )
            return False
        
        # Make room if needed
        while self.current_size_bytes + entry.size_bytes > self.max_size_bytes and self.access_order:
            self._evict_lru()
        
        # Add new entry
        self.cache[key] = entry
        self.current_size_bytes += entry.size_bytes
        self.access_order.append(key)
        
        return True
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if successful, False otherwise
        """
        if key not in self.cache:
            return False
        
        self._remove(key)
        return True
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
        self.current_size_bytes = 0
    
    def _remove(self, key: str) -> None:
        """
        Remove a key from the cache.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            entry = self.cache[key]
            self.current_size_bytes -= entry.size_bytes
            del self.cache[key]
            self._remove_from_access_order(key)
    
    def _evict_lru(self) -> None:
        """Evict the least recently used item from the cache."""
        if not self.access_order:
            return
        
        lru_key = self.access_order[0]
        self._remove(lru_key)
    
    def _move_to_front(self, key: str) -> None:
        """
        Move a key to the front of the access order list.
        
        Args:
            key: Cache key
        """
        self._remove_from_access_order(key)
        self.access_order.append(key)
    
    def _remove_from_access_order(self, key: str) -> None:
        """
        Remove a key from the access order list.
        
        Args:
            key: Cache key
        """
        try:
            self.access_order.remove(key)
        except ValueError:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size_bytes": self.current_size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "usage_percent": (self.current_size_bytes / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0,
            "item_count": len(self.cache),
            "items_by_type": self._count_items_by_type()
        }
    
    def _count_items_by_type(self) -> Dict[str, int]:
        """
        Count items by type based on key prefix.
        
        Returns:
            Dictionary with counts by type
        """
        counts = {}
        for key in self.cache:
            prefix = key.split(":")[0] if ":" in key else "other"
            counts[prefix] = counts.get(prefix, 0) + 1
        return counts

# Global in-memory cache instance
memory_cache = InMemoryLRUCache()

async def init_cache() -> None:
    """Initialize the cache service."""
    global VIDEO_METADATA_TTL, TRANSCRIPT_TTL, SUMMARY_TTL
    
    # Load TTL values from config
    settings = get_settings()
    VIDEO_METADATA_TTL = settings.cache_settings.VIDEO_METADATA_TTL
    TRANSCRIPT_TTL = settings.cache_settings.TRANSCRIPT_TTL
    SUMMARY_TTL = settings.cache_settings.SUMMARY_TTL
    
    # Initialize Redis client
    await redis_client.connect_to_redis()
    
    logger.info("Cache service initialized")
    logger.info(f"In-memory cache size: {IN_MEMORY_CACHE_SIZE_BYTES / (1024 * 1024):.2f}MB")
    logger.info(f"Video metadata TTL: {VIDEO_METADATA_TTL}s ({VIDEO_METADATA_TTL / 3600:.1f}h)")
    logger.info(f"Transcript TTL: {TRANSCRIPT_TTL}s ({TRANSCRIPT_TTL / 3600:.1f}h)")
    logger.info(f"Summary TTL: {SUMMARY_TTL}s ({SUMMARY_TTL / 3600:.1f}h)")

async def close_cache() -> None:
    """Close the cache service."""
    await redis_client.close_redis_connection()
    logger.info("Cache service closed")

async def get_cached_value(key: str) -> Optional[Any]:
    """
    Get a value from the cache (multi-level).
    
    This function checks the in-memory cache first, then falls back to Redis.
    
    Args:
        key: Cache key
    
    Returns:
        Cached value or None if not found
    """
    # Check in-memory cache first (faster)
    value = memory_cache.get(key)
    if value is not None:
        logger.debug(f"In-memory cache hit for key: {key}")
        return value
    
    # If not in memory, check Redis
    redis_value = await redis_client.get_cache(key)
    if redis_value is not None:
        logger.debug(f"Redis cache hit for key: {key}")
        
        # Store in memory cache for faster future access
        memory_cache.set(key, redis_value)
        
        return redis_value
    
    logger.debug(f"Cache miss for key: {key}")
    return None

async def set_cached_value(
    key: str,
    value: Any,
    ttl: Optional[int] = None,
    resource_type: Optional[str] = None
) -> bool:
    """
    Set a value in the cache (multi-level).
    
    This function stores the value in both the in-memory cache and Redis.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time-to-live in seconds (None for resource_type-based TTL)
        resource_type: Type of resource (video_metadata, transcript, summary)
    
    Returns:
        True if successful, False otherwise
    """
    # Determine TTL based on resource type if not explicitly provided
    if ttl is None and resource_type:
        if resource_type == "video_metadata":
            ttl = VIDEO_METADATA_TTL
        elif resource_type == "transcript":
            ttl = TRANSCRIPT_TTL
        elif resource_type == "summary":
            ttl = SUMMARY_TTL
    
    # Store in Redis
    redis_success = await redis_client.set_cache(key, value, ttl)
    
    # Store in memory cache
    memory_success = memory_cache.set(key, value, ttl)
    
    return redis_success and memory_success

async def delete_cached_value(key: str) -> bool:
    """
    Delete a value from the cache (multi-level).
    
    This function removes the value from both the in-memory cache and Redis.
    
    Args:
        key: Cache key
    
    Returns:
        True if successful, False otherwise
    """
    # Delete from memory cache
    memory_success = memory_cache.delete(key)
    
    # Delete from Redis
    redis_success = await redis_client.delete_cache(key)
    
    return memory_success or redis_success

async def clear_cache() -> bool:
    """
    Clear all cache (multi-level).
    
    Returns:
        True if successful, False otherwise
    """
    # Clear memory cache
    memory_cache.clear()
    
    # Clear Redis cache
    redis_success = await redis_client.clear_cache()
    
    return redis_success

async def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    # Get Redis stats
    redis_stats = await redis_client.get_cache_stats()
    
    # Get memory cache stats
    memory_stats = memory_cache.get_stats()
    
    return {
        "redis": redis_stats,
        "memory": memory_stats
    }

# Specific cache operations for YouTube Summarizer

async def cache_video_info(video_id: str, video_info: Dict[str, Any]) -> bool:
    """
    Cache video information.
    
    Args:
        video_id: YouTube video ID
        video_info: Dictionary containing video information
    
    Returns:
        True if successful, False otherwise
    """
    key = VIDEO_INFO_KEY_PATTERN.format(video_id=video_id)
    return await set_cached_value(key, video_info, resource_type="video_metadata")

async def get_cached_video_info(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached video information.
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        Dictionary containing video information or None if not cached
    """
    key = VIDEO_INFO_KEY_PATTERN.format(video_id=video_id)
    return await get_cached_value(key)

async def cache_transcript(video_id: str, transcript_data: Dict[str, Any]) -> bool:
    """
    Cache transcript data for a video.
    
    Args:
        video_id: YouTube video ID
        transcript_data: Dictionary containing transcript text and language
    
    Returns:
        True if successful, False otherwise
    """
    key = TRANSCRIPT_KEY_PATTERN.format(video_id=video_id)
    return await set_cached_value(key, transcript_data, resource_type="transcript")

async def get_cached_transcript(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached transcript data for a video.
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        Dictionary containing transcript text and language or None if not cached
    """
    key = TRANSCRIPT_KEY_PATTERN.format(video_id=video_id)
    return await get_cached_value(key)

async def cache_summary_result(
    video_id: str,
    summary_type: str,
    summary_length: str,
    summary_text: str
) -> bool:
    """
    Cache a generated summary result.
    
    Args:
        video_id: YouTube video ID
        summary_type: Type of summary (Brief, Detailed, etc.)
        summary_length: Length of summary (Short, Medium, Long)
        summary_text: The generated summary text
    
    Returns:
        True if successful, False otherwise
    """
    key = SUMMARY_KEY_PATTERN.format(
        video_id=video_id,
        summary_type=summary_type,
        summary_length=summary_length
    )
    
    # Store summary data
    summary_data = {
        'video_id': video_id,
        'summary_type': summary_type,
        'summary_length': summary_length,
        'summary_text': summary_text,
        'generated_at': datetime.now(timezone.utc).isoformat()
    }
    
    return await set_cached_value(key, summary_data, resource_type="summary")

async def get_cached_summary(
    video_id: str,
    summary_type: str,
    summary_length: str
) -> Optional[Dict[str, Any]]:
    """
    Get a cached summary result.
    
    Args:
        video_id: YouTube video ID
        summary_type: Type of summary (Brief, Detailed, etc.)
        summary_length: Length of summary (Short, Medium, Long)
    
    Returns:
        Dictionary containing summary data or None if not cached
    """
    key = SUMMARY_KEY_PATTERN.format(
        video_id=video_id,
        summary_type=summary_type,
        summary_length=summary_length
    )
    return await get_cached_value(key)

async def cache_available_languages(video_id: str, languages: Dict[str, Any]) -> bool:
    """
    Cache available subtitle languages for a video.
    
    Args:
        video_id: YouTube video ID
        languages: Dictionary containing available languages
    
    Returns:
        True if successful, False otherwise
    """
    key = LANGUAGES_KEY_PATTERN.format(video_id=video_id)
    return await set_cached_value(key, languages, resource_type="video_metadata")

async def get_cached_languages(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached available subtitle languages for a video.
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        Dictionary containing available languages or None if not cached
    """
    key = LANGUAGES_KEY_PATTERN.format(video_id=video_id)
    return await get_cached_value(key)

async def invalidate_video_cache(
    video_id: str,
    invalidate_all: bool = False,
    invalidate_types: Optional[List[str]] = None
) -> List[str]:
    """
    Invalidate cache for a video.
    
    Args:
        video_id: YouTube video ID
        invalidate_all: Whether to invalidate all caches for the video
        invalidate_types: Specific cache types to invalidate
    
    Returns:
        List of invalidated cache types
    """
    invalidated = []
    
    if invalidate_all or (invalidate_types and "video_info" in invalidate_types):
        key = VIDEO_INFO_KEY_PATTERN.format(video_id=video_id)
        if await delete_cached_value(key):
            invalidated.append("video_info")
    
    if invalidate_all or (invalidate_types and "transcript" in invalidate_types):
        key = TRANSCRIPT_KEY_PATTERN.format(video_id=video_id)
        if await delete_cached_value(key):
            invalidated.append("transcript")
    
    if invalidate_all or (invalidate_types and "summary" in invalidate_types):
        # Get all summary keys for this video
        pattern = SUMMARY_KEY_PATTERN.format(
            video_id=video_id,
            summary_type="*",
            summary_length="*"
        )
        summary_keys = await redis_client.get_keys_by_pattern(pattern)
        
        # Delete each summary key
        for key in summary_keys:
            await delete_cached_value(key)
        
        if summary_keys:
            invalidated.append("summary")
    
    if invalidate_all or (invalidate_types and "languages" in invalidate_types):
        key = LANGUAGES_KEY_PATTERN.format(video_id=video_id)
        if await delete_cached_value(key):
            invalidated.append("languages")
    
    return invalidated
