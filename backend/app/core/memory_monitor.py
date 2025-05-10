"""
Memory monitoring utility for the YouTube Summarizer API.

This module provides functionality to monitor memory usage and log it at regular intervals.
It also provides functions to check if memory usage exceeds certain thresholds.
"""

import os
import psutil
import logging
import asyncio
from typing import Dict, Any, Optional, Callable
import time
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring utility class."""
    
    def __init__(
        self,
        interval_seconds: int = 60,
        warning_threshold_mb: int = 410,  # 80% of 512MB
        critical_threshold_mb: int = 460,  # 90% of 512MB
        warning_callback: Optional[Callable] = None,
        critical_callback: Optional[Callable] = None,
        enabled: bool = True
    ):
        """
        Initialize the memory monitor.
        
        Args:
            interval_seconds: Interval between memory checks in seconds
            warning_threshold_mb: Memory threshold for warning in MB
            critical_threshold_mb: Memory threshold for critical alert in MB
            warning_callback: Function to call when warning threshold is exceeded
            critical_callback: Function to call when critical threshold is exceeded
            enabled: Whether the monitor is enabled
        """
        self.interval_seconds = interval_seconds
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.warning_callback = warning_callback
        self.critical_callback = critical_callback
        self.enabled = enabled
        self.running = False
        self.task = None
        self.history = []  # Store recent memory measurements
        self.max_history_size = 60  # Keep last 60 measurements (1 hour at 1-minute interval)
        
        # Process ID for monitoring
        self.process = psutil.Process(os.getpid())
    
    async def start(self) -> None:
        """Start the memory monitoring task."""
        if not self.enabled:
            logger.info("Memory monitoring is disabled")
            return
        
        if self.running:
            logger.warning("Memory monitor is already running")
            return
        
        self.running = True
        self.task = asyncio.create_task(self._monitor_task())
        logger.info(f"Memory monitoring started with {self.interval_seconds}s interval")
    
    async def stop(self) -> None:
        """Stop the memory monitoring task."""
        if not self.running or not self.task:
            return
        
        self.running = False
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        logger.info("Memory monitoring stopped")
    
    async def _monitor_task(self) -> None:
        """Background task that monitors memory usage."""
        while self.running:
            try:
                memory_info = self.get_memory_info()
                self._log_memory_usage(memory_info)
                self._check_thresholds(memory_info)
                
                # Store in history
                self.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "memory_mb": memory_info["memory_mb"],
                    "percent": memory_info["percent"]
                })
                
                # Trim history if needed
                if len(self.history) > self.max_history_size:
                    self.history = self.history[-self.max_history_size:]
                
                # Wait for next check
                await asyncio.sleep(self.interval_seconds)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(self.interval_seconds)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get current memory usage information.
        
        Returns:
            Dict with memory usage information
        """
        try:
            # Get memory info for the current process
            process_memory = self.process.memory_info()
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            # Calculate memory usage
            memory_mb = process_memory.rss / (1024 * 1024)  # Convert to MB
            percent = (memory_mb / (system_memory.total / (1024 * 1024))) * 100
            
            return {
                "timestamp": time.time(),
                "memory_mb": memory_mb,
                "memory_bytes": process_memory.rss,
                "percent": percent,
                "system_total_mb": system_memory.total / (1024 * 1024),
                "system_available_mb": system_memory.available / (1024 * 1024),
                "system_percent": system_memory.percent
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {
                "timestamp": time.time(),
                "memory_mb": 0,
                "memory_bytes": 0,
                "percent": 0,
                "system_total_mb": 0,
                "system_available_mb": 0,
                "system_percent": 0,
                "error": str(e)
            }
    
    def _log_memory_usage(self, memory_info: Dict[str, Any]) -> None:
        """
        Log memory usage information.
        
        Args:
            memory_info: Memory usage information
        """
        logger.info(
            f"Memory usage: {memory_info['memory_mb']:.2f}MB "
            f"({memory_info['percent']:.2f}% of system memory)"
        )
    
    def _check_thresholds(self, memory_info: Dict[str, Any]) -> None:
        """
        Check if memory usage exceeds thresholds and trigger callbacks if needed.
        
        Args:
            memory_info: Memory usage information
        """
        memory_mb = memory_info["memory_mb"]
        
        # Check critical threshold first
        if memory_mb >= self.critical_threshold_mb:
            logger.warning(
                f"CRITICAL: Memory usage ({memory_mb:.2f}MB) "
                f"exceeds critical threshold ({self.critical_threshold_mb}MB)"
            )
            if self.critical_callback:
                try:
                    self.critical_callback(memory_info)
                except Exception as e:
                    logger.error(f"Error in critical memory callback: {e}")
        
        # Check warning threshold
        elif memory_mb >= self.warning_threshold_mb:
            logger.warning(
                f"WARNING: Memory usage ({memory_mb:.2f}MB) "
                f"exceeds warning threshold ({self.warning_threshold_mb}MB)"
            )
            if self.warning_callback:
                try:
                    self.warning_callback(memory_info)
                except Exception as e:
                    logger.error(f"Error in warning memory callback: {e}")
    
    def is_memory_critical(self) -> bool:
        """
        Check if memory usage is currently critical.
        
        Returns:
            True if memory usage exceeds critical threshold, False otherwise
        """
        memory_info = self.get_memory_info()
        return memory_info["memory_mb"] >= self.critical_threshold_mb
    
    def is_memory_warning(self) -> bool:
        """
        Check if memory usage is currently at warning level.
        
        Returns:
            True if memory usage exceeds warning threshold, False otherwise
        """
        memory_info = self.get_memory_info()
        return memory_info["memory_mb"] >= self.warning_threshold_mb
    
    def get_memory_history(self) -> list:
        """
        Get memory usage history.
        
        Returns:
            List of memory usage measurements
        """
        return self.history

# Global memory monitor instance
memory_monitor = None

def init_memory_monitor(
    interval_seconds: int = 60,
    warning_threshold_mb: int = 410,
    critical_threshold_mb: int = 460,
    warning_callback: Optional[Callable] = None,
    critical_callback: Optional[Callable] = None,
    enabled: bool = True
) -> MemoryMonitor:
    """
    Initialize the global memory monitor.
    
    Args:
        interval_seconds: Interval between memory checks in seconds
        warning_threshold_mb: Memory threshold for warning in MB
        critical_threshold_mb: Memory threshold for critical alert in MB
        warning_callback: Function to call when warning threshold is exceeded
        critical_callback: Function to call when critical threshold is exceeded
        enabled: Whether the monitor is enabled
    
    Returns:
        MemoryMonitor instance
    """
    global memory_monitor
    memory_monitor = MemoryMonitor(
        interval_seconds=interval_seconds,
        warning_threshold_mb=warning_threshold_mb,
        critical_threshold_mb=critical_threshold_mb,
        warning_callback=warning_callback,
        critical_callback=critical_callback,
        enabled=enabled
    )
    return memory_monitor

def get_memory_monitor() -> Optional[MemoryMonitor]:
    """
    Get the global memory monitor instance.
    
    Returns:
        MemoryMonitor instance or None if not initialized
    """
    return memory_monitor
