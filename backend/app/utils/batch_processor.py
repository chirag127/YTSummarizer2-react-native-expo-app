"""
Batch processor utility for the YouTube Summarizer API.

This module provides functionality for batching requests to optimize
resource usage in resource-constrained environments.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, Callable, TypeVar, Generic, Awaitable
from dataclasses import dataclass, field

from ..core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic function signatures
T = TypeVar('T')
R = TypeVar('R')

@dataclass
class BatchItem(Generic[T, R]):
    """
    Class representing an item in a batch.
    
    Attributes:
        data: Input data for the batch item
        future: Future to be resolved with the result
        created_at: Creation timestamp
    """
    data: T
    future: asyncio.Future[R]
    created_at: float = field(default_factory=time.time)

class BatchProcessor(Generic[T, R]):
    """
    Batch processor for grouping similar requests.
    
    This class provides functionality to:
    1. Collect similar requests into batches
    2. Process batches when they reach a certain size or age
    3. Distribute results back to individual requesters
    """
    
    def __init__(
        self,
        name: str,
        processor_func: Callable[[List[T]], Awaitable[List[R]]],
        max_batch_size: int = 5,
        max_wait_time: float = 2.0
    ):
        """
        Initialize the batch processor.
        
        Args:
            name: Processor name for identification
            processor_func: Function to process a batch of items
            max_batch_size: Maximum number of items in a batch
            max_wait_time: Maximum time to wait before processing a batch (seconds)
        """
        self.name = name
        self.processor_func = processor_func
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.current_batch: List[BatchItem[T, R]] = []
        self.batch_lock = asyncio.Lock()
        self.processing_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self.total_processed = 0
        self.total_batches = 0
    
    async def start(self) -> None:
        """Start the batch processor."""
        if self.processing_task is not None:
            logger.warning(f"Batch processor '{self.name}' is already running")
            return
        
        self.shutdown_event.clear()
        self.processing_task = asyncio.create_task(self._processing_loop())
        logger.info(f"Batch processor '{self.name}' started")
    
    async def stop(self) -> None:
        """Stop the batch processor."""
        if self.processing_task is None:
            logger.warning(f"Batch processor '{self.name}' is not running")
            return
        
        self.shutdown_event.set()
        await self.processing_task
        self.processing_task = None
        logger.info(f"Batch processor '{self.name}' stopped")
        
        # Process any remaining items
        await self._process_current_batch()
    
    async def add_item(self, data: T) -> R:
        """
        Add an item to the batch.
        
        Args:
            data: Input data for the item
        
        Returns:
            Result of processing the item
        """
        # Create a future to be resolved when the batch is processed
        future: asyncio.Future[R] = asyncio.Future()
        
        # Create a batch item
        item = BatchItem(data=data, future=future)
        
        # Add the item to the current batch
        async with self.batch_lock:
            self.current_batch.append(item)
            
            # If the batch is full, trigger processing
            if len(self.current_batch) >= self.max_batch_size:
                # Process in a separate task to avoid blocking
                asyncio.create_task(self._process_current_batch())
        
        # Wait for the result
        return await future
    
    async def _processing_loop(self) -> None:
        """Background task that processes batches periodically."""
        while not self.shutdown_event.is_set():
            # Check if we need to process the current batch
            async with self.batch_lock:
                if self.current_batch:
                    oldest_item = min(self.current_batch, key=lambda item: item.created_at)
                    age = time.time() - oldest_item.created_at
                    
                    if age >= self.max_wait_time:
                        # Process in a separate task to avoid blocking
                        asyncio.create_task(self._process_current_batch())
            
            # Wait a short time before checking again
            try:
                await asyncio.wait_for(
                    self.shutdown_event.wait(),
                    timeout=0.1  # 100ms
                )
            except asyncio.TimeoutError:
                pass
    
    async def _process_current_batch(self) -> None:
        """Process the current batch of items."""
        # Get the current batch
        async with self.batch_lock:
            if not self.current_batch:
                return
            
            batch_to_process = self.current_batch
            self.current_batch = []
        
        # Log batch processing
        logger.info(f"Processing batch of {len(batch_to_process)} items in '{self.name}'")
        
        try:
            # Extract data from batch items
            batch_data = [item.data for item in batch_to_process]
            
            # Process the batch
            results = await self.processor_func(batch_data)
            
            # Check if we have the correct number of results
            if len(results) != len(batch_to_process):
                logger.error(
                    f"Batch processor '{self.name}' returned {len(results)} results "
                    f"for {len(batch_to_process)} items"
                )
                
                # Resolve futures with errors
                for item in batch_to_process:
                    if not item.future.done():
                        item.future.set_exception(
                            RuntimeError("Batch processing failed: incorrect number of results")
                        )
                
                return
            
            # Resolve futures with results
            for item, result in zip(batch_to_process, results):
                if not item.future.done():
                    item.future.set_result(result)
            
            # Update statistics
            self.total_processed += len(batch_to_process)
            self.total_batches += 1
        except Exception as e:
            logger.error(f"Error processing batch in '{self.name}': {e}")
            
            # Resolve futures with the error
            for item in batch_to_process:
                if not item.future.done():
                    item.future.set_exception(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns:
            Dictionary with processor statistics
        """
        return {
            "name": self.name,
            "max_batch_size": self.max_batch_size,
            "max_wait_time": self.max_wait_time,
            "current_batch_size": len(self.current_batch),
            "total_processed": self.total_processed,
            "total_batches": self.total_batches,
            "average_batch_size": (
                self.total_processed / self.total_batches
                if self.total_batches > 0 else 0
            ),
            "is_running": self.processing_task is not None
        }

# Global batch processor registry
_batch_processors: Dict[str, BatchProcessor] = {}

def get_batch_processor(name: str) -> Optional[BatchProcessor]:
    """
    Get a batch processor by name.
    
    Args:
        name: Batch processor name
    
    Returns:
        Batch processor or None if not found
    """
    return _batch_processors.get(name)

def register_batch_processor(
    name: str,
    processor_func: Callable[[List[Any]], Awaitable[List[Any]]],
    max_batch_size: int = 5,
    max_wait_time: float = 2.0
) -> BatchProcessor:
    """
    Register a new batch processor.
    
    Args:
        name: Batch processor name
        processor_func: Function to process a batch of items
        max_batch_size: Maximum number of items in a batch
        max_wait_time: Maximum time to wait before processing a batch (seconds)
    
    Returns:
        Batch processor
    """
    if name in _batch_processors:
        return _batch_processors[name]
    
    processor = BatchProcessor(
        name=name,
        processor_func=processor_func,
        max_batch_size=max_batch_size,
        max_wait_time=max_wait_time
    )
    _batch_processors[name] = processor
    return processor

async def init_batch_processors() -> None:
    """Initialize batch processors from configuration."""
    settings = get_settings()
    
    # Start all registered processors
    for processor in _batch_processors.values():
        await processor.start()
    
    logger.info("Batch processors initialized")
