"""
API dependencies for the YouTube Summarizer API.

This module provides common dependencies used across API endpoints.
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import Depends, HTTPException, Request, status

from ..core.config import get_settings
from ..core.middleware import is_degraded_mode
from ..db import mongodb, redis_client
from ..services import cache_service

# Configure logging
logger = logging.getLogger(__name__)

async def get_database():
    """
    Get MongoDB database instance.
    
    Returns:
        MongoDB database
    
    Raises:
        HTTPException: If database connection fails
    """
    try:
        return await mongodb.get_database()
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection error"
        )

async def get_redis():
    """
    Get Redis client instance.
    
    Returns:
        Redis client
    
    Raises:
        HTTPException: If Redis connection fails
    """
    try:
        return await redis_client.get_redis()
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache connection error"
        )

async def check_degraded_mode(request: Request):
    """
    Check if the application is in degraded mode.
    
    Args:
        request: Request object
    
    Raises:
        HTTPException: If in degraded mode and the endpoint is not allowed
    """
    if is_degraded_mode(request):
        # Get current path
        path = request.url.path
        
        # Define allowed paths in degraded mode
        allowed_paths = [
            "/health",
            "/validate-url",
            "/summaries"  # Allow reading summaries but not creating
        ]
        
        # Check if path is allowed
        if not any(path.startswith(allowed_path) for allowed_path in allowed_paths):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is in degraded mode. This endpoint is temporarily unavailable."
            )

async def verify_api_key(x_api_key: Optional[str] = None):
    """
    Verify API key if provided.
    
    Args:
        x_api_key: API key from header
    
    Returns:
        True if valid, False otherwise
    """
    if not x_api_key:
        return False
    
    # In a real implementation, you would verify the API key against a database
    # For now, just check if it's not empty
    return bool(x_api_key)

async def get_memory_stats():
    """
    Get memory usage statistics.
    
    Returns:
        Memory usage statistics
    """
    from ..core.memory_monitor import get_memory_monitor
    
    memory_monitor = get_memory_monitor()
    if not memory_monitor:
        return {
            "memory_mb": 0,
            "percent": 0,
            "warning": False,
            "critical": False
        }
    
    memory_info = memory_monitor.get_memory_info()
    return {
        "memory_mb": memory_info["memory_mb"],
        "percent": memory_info["percent"],
        "warning": memory_monitor.is_memory_warning(),
        "critical": memory_monitor.is_memory_critical()
    }
