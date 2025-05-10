"""
Main FastAPI application for the YouTube Summarizer API.

This module initializes the FastAPI application with all middleware,
routes, and dependencies optimized for resource-constrained environments.
"""

import os
import time
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Callable

from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import get_settings, load_environment_settings
from .core.memory_monitor import init_memory_monitor, get_memory_monitor
from .core.middleware import (
    MemoryUsageMiddleware,
    GracefulDegradationMiddleware,
    RequestValidationMiddleware
)
from .db import mongodb, redis_client
from .services import cache_service, youtube_service, gemini_service
from .utils import rate_limiter, request_throttler, batch_processor
from .models.response_models import HealthCheckResponse, MemoryUsageResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Application start time
start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup: Initialize all components
    try:
        # Load environment-specific settings
        settings = get_settings()
        
        # Initialize memory monitor
        memory_monitor = init_memory_monitor(
            interval_seconds=settings.MEMORY_MONITORING_INTERVAL,
            warning_threshold_mb=settings.resource_limits.MEMORY_WARNING_THRESHOLD,
            critical_threshold_mb=settings.resource_limits.MAX_MEMORY_USAGE,
            enabled=settings.ENABLE_MEMORY_MONITORING
        )
        await memory_monitor.start()
        
        # Initialize database connections
        await mongodb.connect_to_mongodb()
        await redis_client.connect_to_redis()
        
        # Initialize cache service
        await cache_service.init_cache()
        
        # Initialize rate limiters
        await rate_limiter.init_rate_limiters()
        
        # Initialize request throttlers
        await request_throttler.init_throttlers()
        
        # Initialize batch processors
        await batch_processor.init_batch_processors()
        
        # Initialize services
        await youtube_service.init_youtube_service()
        await gemini_service.init_gemini_service()
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise
    
    # Yield control back to FastAPI
    yield
    
    # Shutdown: Clean up all resources
    try:
        # Stop memory monitor
        memory_monitor = get_memory_monitor()
        if memory_monitor:
            await memory_monitor.stop()
        
        # Close database connections
        await mongodb.close_mongodb_connection()
        await redis_client.close_redis_connection()
        
        # Close cache service
        await cache_service.close_cache()
        
        # Stop batch processors
        for processor in batch_processor.get_all_batch_processors().values():
            await processor.stop()
        
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Summarizer API",
    description="API for summarizing YouTube videos and answering questions about them",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(
    MemoryUsageMiddleware,
    critical_threshold_mb=get_settings().resource_limits.MAX_MEMORY_USAGE,
    warning_threshold_mb=get_settings().resource_limits.MEMORY_WARNING_THRESHOLD
)

app.add_middleware(
    GracefulDegradationMiddleware,
    warning_threshold_mb=get_settings().resource_limits.MEMORY_WARNING_THRESHOLD
)

app.add_middleware(
    RequestValidationMiddleware,
    max_payload_kb=get_settings().resource_limits.MAX_REQUEST_PAYLOAD_KB,
    max_url_length=get_settings().resource_limits.MAX_URL_LENGTH,
    request_timeout=get_settings().resource_limits.REQUEST_TIMEOUT_SECONDS
)

# Add rate limiting middleware
app.add_middleware(
    rate_limiter.RateLimiterMiddleware,
    global_rate_limit=get_settings().resource_limits.GLOBAL_RATE_LIMIT_PER_MINUTE,
    ip_rate_limit=get_settings().resource_limits.IP_RATE_LIMIT_PER_MINUTE
)

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the application
    """
    # Get memory usage
    memory_monitor = get_memory_monitor()
    memory_usage = memory_monitor.get_memory_info() if memory_monitor else {
        "timestamp": time.time(),
        "memory_mb": 0,
        "memory_bytes": 0,
        "percent": 0,
        "system_total_mb": 0,
        "system_available_mb": 0,
        "system_percent": 0
    }
    
    # Check database connection
    db_status = "Connected"
    try:
        db = await mongodb.get_database()
        await db.command("ping")
    except Exception as e:
        db_status = f"Error: {str(e)}"
    
    # Check Redis connection
    cache_status = "Connected"
    try:
        redis = await redis_client.get_redis()
        await redis.ping()
    except Exception as e:
        cache_status = f"Error: {str(e)}"
    
    # Check if in degraded mode
    degraded_mode = False
    if memory_monitor:
        degraded_mode = memory_monitor.is_memory_warning()
    
    return HealthCheckResponse(
        status="ok",
        version="1.0.0",
        timestamp=time.time(),
        uptime=time.time() - start_time,
        memory_usage=MemoryUsageResponse(**memory_usage),
        database_status=db_status,
        cache_status=cache_status,
        degraded_mode=degraded_mode
    )

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions.
    
    Args:
        request: The request that caused the exception
        exc: The exception
    
    Returns:
        JSON response with error details
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions.
    
    Args:
        request: The request that caused the exception
        exc: The exception
    
    Returns:
        JSON response with error details
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred"}
    )

@app.exception_handler(request_throttler.ThrottlingError)
async def throttling_exception_handler(request: Request, exc: request_throttler.ThrottlingError):
    """
    Handle throttling exceptions.
    
    Args:
        request: The request that caused the exception
        exc: The exception
    
    Returns:
        JSON response with error details
    """
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": str(exc),
            "throttler": exc.throttler_name
        },
        headers={"Retry-After": "30"}
    )

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns:
        Welcome message
    """
    return {"message": "YouTube Summarizer API is running"}

# Import and include API routers
# This is done at the end to avoid circular imports
from .api.endpoints import summaries, validation, qa

app.include_router(validation.router, tags=["validation"])
app.include_router(summaries.router, tags=["summaries"])
app.include_router(qa.router, tags=["qa"])
