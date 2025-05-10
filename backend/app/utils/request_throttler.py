"""
Request throttler utility for the YouTube Summarizer API.

This module provides request throttling functionality to limit the number
of concurrent requests for resource-intensive operations.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Callable, TypeVar, Awaitable
from functools import wraps

from ..core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic function signatures
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Awaitable[Any]])

class RequestThrottler:
    """
    Request throttler for limiting concurrent requests.
    
    This class provides a semaphore-based throttler that:
    1. Limits the number of concurrent requests for a specific operation
    2. Queues excess requests up to a maximum queue size
    3. Rejects requests when the queue is full
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent: int,
        max_queue_size: Optional[int] = None
    ):
        """
        Initialize the request throttler.
        
        Args:
            name: Throttler name for identification
            max_concurrent: Maximum number of concurrent requests
            max_queue_size: Maximum queue size (None for unlimited)
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue_size = 0
        self.active_requests = 0
        self.total_requests = 0
        self.rejected_requests = 0
    
    async def acquire(self) -> bool:
        """
        Acquire a slot for a request.
        
        Returns:
            True if slot acquired, False if rejected
        """
        # Check if queue is full
        if self.max_queue_size is not None and self.queue_size >= self.max_queue_size:
            self.rejected_requests += 1
            logger.warning(
                f"Request rejected by throttler '{self.name}': "
                f"queue full ({self.queue_size}/{self.max_queue_size})"
            )
            return False
        
        # Increment queue size
        self.queue_size += 1
        self.total_requests += 1
        
        # Acquire semaphore
        await self.semaphore.acquire()
        
        # Update counters
        self.queue_size -= 1
        self.active_requests += 1
        
        return True
    
    def release(self) -> None:
        """Release a slot after a request is completed."""
        self.active_requests -= 1
        self.semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get throttler statistics.
        
        Returns:
            Dictionary with throttler statistics
        """
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "active_requests": self.active_requests,
            "queue_size": self.queue_size,
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "rejection_rate": (
                self.rejected_requests / self.total_requests * 100
                if self.total_requests > 0 else 0
            )
        }

class ThrottlingError(Exception):
    """Exception raised when a request is rejected by a throttler."""
    
    def __init__(self, throttler_name: str):
        """
        Initialize the exception.
        
        Args:
            throttler_name: Name of the throttler that rejected the request
        """
        self.throttler_name = throttler_name
        super().__init__(f"Request rejected by throttler '{throttler_name}': too many concurrent requests")

def throttled(throttler_name: str) -> Callable[[F], F]:
    """
    Decorator for throttling async functions.
    
    Args:
        throttler_name: Name of the throttler to use
    
    Returns:
        Decorated function
    
    Raises:
        ThrottlingError: If the request is rejected by the throttler
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            throttler = get_throttler(throttler_name)
            if not throttler:
                logger.warning(f"Throttler '{throttler_name}' not found, proceeding without throttling")
                return await func(*args, **kwargs)
            
            # Try to acquire a slot
            if not await throttler.acquire():
                raise ThrottlingError(throttler_name)
            
            try:
                # Execute the function
                return await func(*args, **kwargs)
            finally:
                # Release the slot
                throttler.release()
        
        return wrapper  # type: ignore
    
    return decorator

# Global throttler registry
_throttlers: Dict[str, RequestThrottler] = {}

def get_throttler(name: str) -> Optional[RequestThrottler]:
    """
    Get a throttler by name.
    
    Args:
        name: Throttler name
    
    Returns:
        Throttler or None if not found
    """
    return _throttlers.get(name)

def register_throttler(
    name: str,
    max_concurrent: int,
    max_queue_size: Optional[int] = None
) -> RequestThrottler:
    """
    Register a new throttler.
    
    Args:
        name: Throttler name
        max_concurrent: Maximum number of concurrent requests
        max_queue_size: Maximum queue size (None for unlimited)
    
    Returns:
        Throttler
    """
    if name in _throttlers:
        return _throttlers[name]
    
    throttler = RequestThrottler(
        name=name,
        max_concurrent=max_concurrent,
        max_queue_size=max_queue_size
    )
    _throttlers[name] = throttler
    return throttler

def get_all_throttlers() -> Dict[str, RequestThrottler]:
    """
    Get all registered throttlers.
    
    Returns:
        Dictionary of throttlers
    """
    return _throttlers.copy()

async def init_throttlers() -> None:
    """Initialize throttlers from configuration."""
    settings = get_settings()
    
    # Register video processing throttler
    register_throttler(
        name="video_processing",
        max_concurrent=settings.resource_limits.VIDEO_PROCESSING_MAX_CONCURRENT,
        max_queue_size=5  # Allow up to 5 requests in queue
    )
    
    # Register AI summarization throttler
    register_throttler(
        name="ai_summarization",
        max_concurrent=settings.resource_limits.AI_SUMMARIZATION_MAX_CONCURRENT,
        max_queue_size=3  # Allow up to 3 requests in queue
    )
    
    # Register video QA throttler
    register_throttler(
        name="video_qa",
        max_concurrent=2,  # Allow 2 concurrent QA requests
        max_queue_size=5   # Allow up to 5 requests in queue
    )
    
    logger.info("Request throttlers initialized")
