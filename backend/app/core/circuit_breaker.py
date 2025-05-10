"""
Circuit breaker implementation for the YouTube Summarizer API.

This module provides a circuit breaker pattern implementation to prevent
cascading failures when external services are unavailable or experiencing issues.
"""

import time
import logging
import asyncio
from enum import Enum
from typing import Callable, Any, Dict, Optional, TypeVar, Awaitable
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic function signatures
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests are allowed
    OPEN = "open"          # Circuit is open, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service is back, limited requests allowed


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.
    
    The circuit breaker has three states:
    - CLOSED: Normal operation, all requests are allowed
    - OPEN: Circuit is open, all requests are rejected
    - HALF_OPEN: Testing if service is back, limited requests are allowed
    
    When the failure count exceeds the threshold, the circuit opens.
    After the recovery time, the circuit goes to half-open state.
    If a request succeeds in half-open state, the circuit closes.
    If a request fails in half-open state, the circuit opens again.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_time: int = 60,
        half_open_timeout: int = 30,
        exception_types: tuple = (Exception,),
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker (for logging)
            failure_threshold: Number of consecutive failures before opening the circuit
            recovery_time: Time in seconds to wait before transitioning to half-open state
            half_open_timeout: Time in seconds between half-open retry attempts
            exception_types: Tuple of exception types that count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.half_open_timeout = half_open_timeout
        self.exception_types = exception_types
        
        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.last_success_time = time.time()
        self.last_state_change_time = time.time()
        self.half_open_next_attempt = 0
    
    def get_state(self) -> CircuitState:
        """
        Get the current state of the circuit breaker.
        
        Returns:
            Current circuit state
        """
        # Check if it's time to transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_time:
                self._transition_to_half_open()
        
        return self.state
    
    def _transition_to_open(self) -> None:
        """Transition the circuit to OPEN state."""
        if self.state != CircuitState.OPEN:
            logger.warning(f"Circuit breaker '{self.name}' is now OPEN")
            self.state = CircuitState.OPEN
            self.last_state_change_time = time.time()
    
    def _transition_to_half_open(self) -> None:
        """Transition the circuit to HALF_OPEN state."""
        if self.state != CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker '{self.name}' is now HALF_OPEN")
            self.state = CircuitState.HALF_OPEN
            self.half_open_next_attempt = time.time()
            self.last_state_change_time = time.time()
    
    def _transition_to_closed(self) -> None:
        """Transition the circuit to CLOSED state."""
        if self.state != CircuitState.CLOSED:
            logger.info(f"Circuit breaker '{self.name}' is now CLOSED")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_state_change_time = time.time()
    
    def record_success(self) -> None:
        """Record a successful operation."""
        self.last_success_time = time.time()
        
        # If in HALF_OPEN state, a success means the service is back
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_closed()
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
        
        elif self.state == CircuitState.HALF_OPEN:
            # If a request fails in half-open state, go back to open
            self._transition_to_open()
    
    def allow_request(self) -> bool:
        """
        Check if a request should be allowed.
        
        Returns:
            True if the request should be allowed, False otherwise
        """
        current_state = self.get_state()
        
        if current_state == CircuitState.CLOSED:
            return True
        
        elif current_state == CircuitState.OPEN:
            return False
        
        elif current_state == CircuitState.HALF_OPEN:
            # In half-open state, only allow one request per timeout period
            current_time = time.time()
            if current_time >= self.half_open_next_attempt:
                self.half_open_next_attempt = current_time + self.half_open_timeout
                return True
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the circuit breaker.
        
        Returns:
            Dictionary with circuit breaker statistics
        """
        current_time = time.time()
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure": f"{current_time - self.last_failure_time:.2f}s ago" if self.last_failure_time > 0 else "never",
            "last_success": f"{current_time - self.last_success_time:.2f}s ago" if self.last_success_time > 0 else "never",
            "last_state_change": f"{current_time - self.last_state_change_time:.2f}s ago",
            "recovery_time": self.recovery_time,
            "half_open_timeout": self.half_open_timeout,
            "next_retry_allowed": f"{self.half_open_next_attempt - current_time:.2f}s" if self.state == CircuitState.HALF_OPEN and self.half_open_next_attempt > current_time else "now"
        }


class CircuitBreakerError(Exception):
    """Exception raised when a circuit breaker is open."""
    
    def __init__(self, circuit_name: str):
        """
        Initialize the exception.
        
        Args:
            circuit_name: Name of the circuit breaker
        """
        self.circuit_name = circuit_name
        super().__init__(f"Circuit breaker '{circuit_name}' is open")


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_time: int = 60,
    half_open_timeout: int = 30,
    exception_types: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator for synchronous functions to apply circuit breaker pattern.
    
    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of consecutive failures before opening the circuit
        recovery_time: Time in seconds to wait before transitioning to half-open state
        half_open_timeout: Time in seconds between half-open retry attempts
        exception_types: Tuple of exception types that count as failures
    
    Returns:
        Decorated function
    """
    cb = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_time=recovery_time,
        half_open_timeout=half_open_timeout,
        exception_types=exception_types,
    )
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not cb.allow_request():
                raise CircuitBreakerError(cb.name)
            
            try:
                result = func(*args, **kwargs)
                cb.record_success()
                return result
            except cb.exception_types as e:
                cb.record_failure()
                raise
        
        return wrapper  # type: ignore
    
    return decorator


def async_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_time: int = 60,
    half_open_timeout: int = 30,
    exception_types: tuple = (Exception,),
) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator for asynchronous functions to apply circuit breaker pattern.
    
    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of consecutive failures before opening the circuit
        recovery_time: Time in seconds to wait before transitioning to half-open state
        half_open_timeout: Time in seconds between half-open retry attempts
        exception_types: Tuple of exception types that count as failures
    
    Returns:
        Decorated function
    """
    cb = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_time=recovery_time,
        half_open_timeout=half_open_timeout,
        exception_types=exception_types,
    )
    
    def decorator(func: AsyncF) -> AsyncF:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not cb.allow_request():
                raise CircuitBreakerError(cb.name)
            
            try:
                result = await func(*args, **kwargs)
                cb.record_success()
                return result
            except cb.exception_types as e:
                cb.record_failure()
                raise
        
        return wrapper  # type: ignore
    
    return decorator


# Global registry of circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}

def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """
    Get a circuit breaker by name.
    
    Args:
        name: Name of the circuit breaker
    
    Returns:
        CircuitBreaker instance or None if not found
    """
    return _circuit_breakers.get(name)

def register_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_time: int = 60,
    half_open_timeout: int = 30,
    exception_types: tuple = (Exception,),
) -> CircuitBreaker:
    """
    Register a new circuit breaker.
    
    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of consecutive failures before opening the circuit
        recovery_time: Time in seconds to wait before transitioning to half-open state
        half_open_timeout: Time in seconds between half-open retry attempts
        exception_types: Tuple of exception types that count as failures
    
    Returns:
        CircuitBreaker instance
    """
    if name in _circuit_breakers:
        return _circuit_breakers[name]
    
    cb = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_time=recovery_time,
        half_open_timeout=half_open_timeout,
        exception_types=exception_types,
    )
    _circuit_breakers[name] = cb
    return cb

def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """
    Get all registered circuit breakers.
    
    Returns:
        Dictionary of circuit breakers
    """
    return _circuit_breakers.copy()
