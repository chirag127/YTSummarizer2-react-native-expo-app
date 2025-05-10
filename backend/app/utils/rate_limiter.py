"""
Rate limiter utility for the YouTube Summarizer API.

This module provides rate limiting functionality to control request rates
and prevent abuse in resource-constrained environments.
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Callable
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter implementation using the token bucket algorithm.
    
    This class provides a token bucket rate limiter that:
    1. Refills tokens at a constant rate
    2. Allows bursts up to the bucket capacity
    3. Is thread-safe for concurrent access
    """
    
    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        refill_interval: float = 1.0,
        initial_tokens: Optional[int] = None
    ):
        """
        Initialize the rate limiter.
        
        Args:
            capacity: Maximum number of tokens in the bucket
            refill_rate: Number of tokens to add per refill_interval
            refill_interval: Time interval in seconds between refills
            initial_tokens: Initial number of tokens (defaults to capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_interval = refill_interval
        self.tokens = capacity if initial_tokens is None else initial_tokens
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self.lock:
            # Refill tokens based on elapsed time
            self._refill()
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Calculate number of tokens to add
        new_tokens = elapsed * (self.refill_rate / self.refill_interval)
        
        if new_tokens > 0:
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now
    
    async def get_tokens(self) -> float:
        """
        Get the current number of tokens in the bucket.
        
        Returns:
            Current number of tokens
        """
        async with self.lock:
            self._refill()
            return self.tokens

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, reset_after: float):
        """
        Initialize the exception.
        
        Args:
            limit: Rate limit (requests per minute)
            reset_after: Time in seconds until the rate limit resets
        """
        self.limit = limit
        self.reset_after = reset_after
        super().__init__(f"Rate limit exceeded: {limit} requests per minute")

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests.
    
    This middleware applies rate limits based on:
    1. Global rate limit for all requests
    2. Per-IP rate limit to prevent abuse
    3. Per-endpoint rate limit for resource-intensive endpoints
    """
    
    def __init__(
        self,
        app: ASGIApp,
        global_rate_limit: int = 60,  # 60 requests per minute
        ip_rate_limit: int = 10,      # 10 requests per minute per IP
        endpoint_rate_limits: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI application
            global_rate_limit: Global rate limit in requests per minute
            ip_rate_limit: Per-IP rate limit in requests per minute
            endpoint_rate_limits: Dictionary mapping endpoint paths to rate limits
        """
        super().__init__(app)
        
        # Initialize rate limiters
        self.global_limiter = RateLimiter(
            capacity=global_rate_limit,
            refill_rate=global_rate_limit,
            refill_interval=60.0  # 1 minute
        )
        
        # Dictionary of IP-based rate limiters
        self.ip_limiters: Dict[str, RateLimiter] = {}
        self.ip_rate_limit = ip_rate_limit
        
        # Dictionary of endpoint-based rate limiters
        self.endpoint_limiters: Dict[str, RateLimiter] = {}
        self.endpoint_rate_limits = endpoint_rate_limits or {
            # Default endpoint rate limits
            "/generate-summary": 5,  # 5 requests per minute
            "/video-qa": 10,         # 10 requests per minute
            "/validate-url": 20      # 20 requests per minute
        }
        
        # Initialize endpoint limiters
        for endpoint, limit in self.endpoint_rate_limits.items():
            self.endpoint_limiters[endpoint] = RateLimiter(
                capacity=limit,
                refill_rate=limit,
                refill_interval=60.0  # 1 minute
            )
        
        logger.info(f"Rate limiter initialized with global limit: {global_rate_limit} req/min")
        logger.info(f"IP rate limit: {ip_rate_limit} req/min")
        logger.info(f"Endpoint rate limits: {self.endpoint_rate_limits}")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and apply rate limiting.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
        
        Returns:
            The response
        """
        # Skip rate limiting for certain paths
        if self._should_skip_rate_limiting(request.url.path):
            return await call_next(request)
        
        # Apply global rate limit
        if not await self.global_limiter.acquire():
            reset_after = 60.0  # 1 minute
            return self._create_rate_limit_response(
                "Global rate limit exceeded",
                self.global_limiter.capacity,
                reset_after
            )
        
        # Apply IP-based rate limit
        client_ip = self._get_client_ip(request)
        ip_limiter = await self._get_ip_limiter(client_ip)
        
        if not await ip_limiter.acquire():
            reset_after = 60.0  # 1 minute
            return self._create_rate_limit_response(
                f"IP rate limit exceeded for {client_ip}",
                self.ip_rate_limit,
                reset_after
            )
        
        # Apply endpoint-based rate limit if applicable
        endpoint_path = self._get_endpoint_path(request.url.path)
        if endpoint_path in self.endpoint_limiters:
            endpoint_limiter = self.endpoint_limiters[endpoint_path]
            
            if not await endpoint_limiter.acquire():
                reset_after = 60.0  # 1 minute
                return self._create_rate_limit_response(
                    f"Endpoint rate limit exceeded for {endpoint_path}",
                    self.endpoint_rate_limits[endpoint_path],
                    reset_after
                )
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers to the response
        global_tokens = await self.global_limiter.get_tokens()
        ip_tokens = await ip_limiter.get_tokens()
        
        response.headers["X-RateLimit-Limit-Global"] = str(self.global_limiter.capacity)
        response.headers["X-RateLimit-Remaining-Global"] = str(int(global_tokens))
        
        response.headers["X-RateLimit-Limit-IP"] = str(self.ip_rate_limit)
        response.headers["X-RateLimit-Remaining-IP"] = str(int(ip_tokens))
        
        if endpoint_path in self.endpoint_limiters:
            endpoint_tokens = await self.endpoint_limiters[endpoint_path].get_tokens()
            response.headers["X-RateLimit-Limit-Endpoint"] = str(self.endpoint_rate_limits[endpoint_path])
            response.headers["X-RateLimit-Remaining-Endpoint"] = str(int(endpoint_tokens))
        
        return response
    
    def _should_skip_rate_limiting(self, path: str) -> bool:
        """
        Check if rate limiting should be skipped for a path.
        
        Args:
            path: Request path
        
        Returns:
            True if rate limiting should be skipped, False otherwise
        """
        # Skip rate limiting for static files, docs, health checks, etc.
        skip_paths = [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
            "/health",
            "/metrics"
        ]
        
        return any(path.startswith(skip_path) for skip_path in skip_paths)
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get the client IP address from a request.
        
        Args:
            request: The incoming request
        
        Returns:
            Client IP address
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        client_host = request.client.host if request.client else "unknown"
        return client_host
    
    async def _get_ip_limiter(self, ip: str) -> RateLimiter:
        """
        Get or create a rate limiter for an IP address.
        
        Args:
            ip: IP address
        
        Returns:
            Rate limiter for the IP
        """
        if ip not in self.ip_limiters:
            self.ip_limiters[ip] = RateLimiter(
                capacity=self.ip_rate_limit,
                refill_rate=self.ip_rate_limit,
                refill_interval=60.0  # 1 minute
            )
            
            # Clean up old IP limiters periodically
            if len(self.ip_limiters) > 1000:  # Arbitrary limit to prevent memory issues
                await self._clean_up_ip_limiters()
        
        return self.ip_limiters[ip]
    
    async def _clean_up_ip_limiters(self) -> None:
        """Clean up old IP limiters to prevent memory leaks."""
        # Keep the 100 most recently used IP limiters
        sorted_ips = sorted(
            self.ip_limiters.keys(),
            key=lambda ip: self.ip_limiters[ip].last_refill,
            reverse=True
        )
        
        # Remove oldest IP limiters
        for ip in sorted_ips[100:]:
            del self.ip_limiters[ip]
        
        logger.info(f"Cleaned up IP rate limiters, kept {len(self.ip_limiters)} entries")
    
    def _get_endpoint_path(self, path: str) -> str:
        """
        Get the endpoint path from a request path.
        
        Args:
            path: Request path
        
        Returns:
            Endpoint path
        """
        # Extract the endpoint path without query parameters
        endpoint_path = path.split("?")[0]
        
        # Remove trailing slash
        if endpoint_path.endswith("/") and len(endpoint_path) > 1:
            endpoint_path = endpoint_path[:-1]
        
        return endpoint_path
    
    def _create_rate_limit_response(
        self,
        message: str,
        limit: int,
        reset_after: float
    ) -> Response:
        """
        Create a response for rate limit exceeded.
        
        Args:
            message: Error message
            limit: Rate limit
            reset_after: Time in seconds until the rate limit resets
        
        Returns:
            JSON response with rate limit information
        """
        from fastapi.responses import JSONResponse
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": message,
                "limit": limit,
                "reset_after": reset_after
            },
            headers={
                "Retry-After": str(int(reset_after)),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Reset": str(int(time.time() + reset_after))
            }
        )

# Global rate limiter registry
_rate_limiters: Dict[str, RateLimiter] = {}

def get_rate_limiter(name: str) -> Optional[RateLimiter]:
    """
    Get a rate limiter by name.
    
    Args:
        name: Rate limiter name
    
    Returns:
        Rate limiter or None if not found
    """
    return _rate_limiters.get(name)

def register_rate_limiter(
    name: str,
    capacity: int,
    refill_rate: float,
    refill_interval: float = 1.0
) -> RateLimiter:
    """
    Register a new rate limiter.
    
    Args:
        name: Rate limiter name
        capacity: Maximum number of tokens in the bucket
        refill_rate: Number of tokens to add per refill_interval
        refill_interval: Time interval in seconds between refills
    
    Returns:
        Rate limiter
    """
    if name in _rate_limiters:
        return _rate_limiters[name]
    
    limiter = RateLimiter(
        capacity=capacity,
        refill_rate=refill_rate,
        refill_interval=refill_interval
    )
    _rate_limiters[name] = limiter
    return limiter

async def check_rate_limit(
    name: str,
    tokens: int = 1
) -> bool:
    """
    Check if a rate limit allows the requested number of tokens.
    
    Args:
        name: Rate limiter name
        tokens: Number of tokens to acquire
    
    Returns:
        True if rate limit allows, False otherwise
    
    Raises:
        ValueError: If rate limiter not found
    """
    limiter = get_rate_limiter(name)
    if not limiter:
        raise ValueError(f"Rate limiter '{name}' not found")
    
    return await limiter.acquire(tokens)

async def init_rate_limiters() -> None:
    """Initialize rate limiters from configuration."""
    settings = get_settings()
    
    # Register global rate limiter
    register_rate_limiter(
        name="global",
        capacity=settings.resource_limits.GLOBAL_RATE_LIMIT_PER_MINUTE,
        refill_rate=settings.resource_limits.GLOBAL_RATE_LIMIT_PER_MINUTE,
        refill_interval=60.0  # 1 minute
    )
    
    # Register IP rate limiter
    register_rate_limiter(
        name="ip",
        capacity=settings.resource_limits.IP_RATE_LIMIT_PER_MINUTE,
        refill_rate=settings.resource_limits.IP_RATE_LIMIT_PER_MINUTE,
        refill_interval=60.0  # 1 minute
    )
    
    # Register endpoint-specific rate limiters
    register_rate_limiter(
        name="generate_summary",
        capacity=5,  # 5 requests per minute
        refill_rate=5,
        refill_interval=60.0
    )
    
    register_rate_limiter(
        name="video_qa",
        capacity=10,  # 10 requests per minute
        refill_rate=10,
        refill_interval=60.0
    )
    
    register_rate_limiter(
        name="validate_url",
        capacity=20,  # 20 requests per minute
        refill_rate=20,
        refill_interval=60.0
    )
    
    logger.info("Rate limiters initialized")
