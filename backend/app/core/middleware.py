"""
Middleware components for the YouTube Summarizer API.

This module provides middleware for:
1. Request rejection when memory usage is too high
2. Graceful degradation of features when resources are constrained
3. Request validation and sanitization
"""

import time
import logging
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .memory_monitor import get_memory_monitor

# Configure logging
logger = logging.getLogger(__name__)

class MemoryUsageMiddleware(BaseHTTPMiddleware):
    """
    Middleware to monitor memory usage and reject requests when memory is too high.
    
    This middleware checks memory usage before processing each request and:
    1. Rejects requests with a 503 Service Unavailable when memory exceeds critical threshold
    2. Adds a warning header when memory exceeds warning threshold
    """
    
    def __init__(
        self,
        app: ASGIApp,
        critical_threshold_mb: int = 460,  # 90% of 512MB
        warning_threshold_mb: int = 410,   # 80% of 512MB
    ):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI application
            critical_threshold_mb: Memory threshold for rejecting requests
            warning_threshold_mb: Memory threshold for warning
        """
        super().__init__(app)
        self.critical_threshold_mb = critical_threshold_mb
        self.warning_threshold_mb = warning_threshold_mb
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and check memory usage.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
        
        Returns:
            The response
        """
        # Get memory monitor
        memory_monitor = get_memory_monitor()
        if not memory_monitor:
            # If memory monitor is not available, just process the request
            return await call_next(request)
        
        # Check memory usage
        memory_info = memory_monitor.get_memory_info()
        memory_mb = memory_info["memory_mb"]
        
        # If memory usage is critical, reject the request
        if memory_mb >= self.critical_threshold_mb:
            logger.warning(
                f"Request rejected due to high memory usage: {memory_mb:.2f}MB "
                f"(threshold: {self.critical_threshold_mb}MB)"
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "detail": "Service temporarily unavailable due to high load. Please try again later."
                },
                headers={"Retry-After": "30"}  # Suggest client to retry after 30 seconds
            )
        
        # Process the request
        response = await call_next(request)
        
        # If memory usage is at warning level, add a warning header
        if memory_mb >= self.warning_threshold_mb:
            response.headers["X-Memory-Warning"] = "High memory usage detected"
        
        return response

class GracefulDegradationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for graceful degradation of features when resources are constrained.
    
    This middleware:
    1. Disables non-essential features when memory exceeds warning threshold
    2. Adds a header to indicate degraded mode
    """
    
    def __init__(
        self,
        app: ASGIApp,
        warning_threshold_mb: int = 410,  # 80% of 512MB
    ):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI application
            warning_threshold_mb: Memory threshold for degrading features
        """
        super().__init__(app)
        self.warning_threshold_mb = warning_threshold_mb
        self.degraded_mode = False
        self.last_check_time = 0
        self.check_interval = 10  # Check memory every 10 seconds
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and apply graceful degradation if needed.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
        
        Returns:
            The response
        """
        # Check if it's time to update degraded mode status
        current_time = time.time()
        if current_time - self.last_check_time > self.check_interval:
            self._update_degraded_mode()
            self.last_check_time = current_time
        
        # Add degraded mode to request state
        request.state.degraded_mode = self.degraded_mode
        
        # Process the request
        response = await call_next(request)
        
        # Add header if in degraded mode
        if self.degraded_mode:
            response.headers["X-Degraded-Mode"] = "true"
        
        return response
    
    def _update_degraded_mode(self) -> None:
        """Update the degraded mode status based on current memory usage."""
        memory_monitor = get_memory_monitor()
        if not memory_monitor:
            return
        
        memory_info = memory_monitor.get_memory_info()
        memory_mb = memory_info["memory_mb"]
        
        # Update degraded mode status
        new_degraded_mode = memory_mb >= self.warning_threshold_mb
        
        # Log if status changed
        if new_degraded_mode != self.degraded_mode:
            if new_degraded_mode:
                logger.warning(
                    f"Entering degraded mode due to high memory usage: {memory_mb:.2f}MB "
                    f"(threshold: {self.warning_threshold_mb}MB)"
                )
            else:
                logger.info(
                    f"Exiting degraded mode, memory usage: {memory_mb:.2f}MB "
                    f"(threshold: {self.warning_threshold_mb}MB)"
                )
        
        self.degraded_mode = new_degraded_mode

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request validation and sanitization.
    
    This middleware:
    1. Validates request payload size
    2. Validates URL length
    3. Adds timeout to requests
    """
    
    def __init__(
        self,
        app: ASGIApp,
        max_payload_kb: int = 10,
        max_url_length: int = 2048,
        request_timeout: int = 60,
    ):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI application
            max_payload_kb: Maximum request payload size in KB
            max_url_length: Maximum URL length in characters
            request_timeout: Request timeout in seconds
        """
        super().__init__(app)
        self.max_payload_kb = max_payload_kb
        self.max_url_length = max_url_length
        self.request_timeout = request_timeout
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and validate it.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
        
        Returns:
            The response
        """
        # Validate URL length
        url_length = len(str(request.url))
        if url_length > self.max_url_length:
            logger.warning(f"Request rejected due to URL length: {url_length} (max: {self.max_url_length})")
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": f"URL length exceeds maximum allowed ({self.max_url_length} characters)"}
            )
        
        # Validate payload size for POST/PUT/PATCH requests
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length:
                content_length_kb = int(content_length) / 1024
                if content_length_kb > self.max_payload_kb:
                    logger.warning(
                        f"Request rejected due to payload size: {content_length_kb:.2f}KB "
                        f"(max: {self.max_payload_kb}KB)"
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": f"Request payload exceeds maximum allowed ({self.max_payload_kb}KB)"}
                    )
        
        # Add timeout to request state
        request.state.timeout = self.request_timeout
        
        # Process the request with timeout
        try:
            # Start timer
            start_time = time.time()
            
            # Process request
            response = await call_next(request)
            
            # Check if request took too long
            elapsed_time = time.time() - start_time
            if elapsed_time > self.request_timeout:
                logger.warning(
                    f"Request completed but exceeded timeout: {elapsed_time:.2f}s "
                    f"(timeout: {self.request_timeout}s)"
                )
                response.headers["X-Request-Timeout-Warning"] = "true"
            
            return response
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "An error occurred while processing the request"}
            )

def get_request_ip(request: Request) -> str:
    """
    Get the client IP address from a request.
    
    Args:
        request: The incoming request
    
    Returns:
        The client IP address
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def is_degraded_mode(request: Request) -> bool:
    """
    Check if the application is in degraded mode.
    
    Args:
        request: The incoming request
    
    Returns:
        True if in degraded mode, False otherwise
    """
    return getattr(request.state, "degraded_mode", False)
