"""
Response models for the YouTube Summarizer API.

This module defines Pydantic models for API responses with
optimized field definitions for resource-constrained environments.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from .request_models import ChatMessage, SummaryType, SummaryLength

class Summary(BaseModel):
    """
    Model for summary data.
    
    Attributes:
        id: Summary ID
        video_url: YouTube video URL
        video_title: Video title
        video_thumbnail_url: Video thumbnail URL
        summary_text: Generated summary text
        summary_type: Type of summary
        summary_length: Length of summary
        transcript_language: Language of the transcript
        is_starred: Whether the summary is starred
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    id: Optional[str] = None
    video_url: str
    video_title: Optional[str] = None
    video_thumbnail_url: Optional[str] = None
    summary_text: str
    summary_type: str
    summary_length: str
    transcript_language: Optional[str] = None
    is_starred: Optional[bool] = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class SummaryResponse(BaseModel):
    """
    Model for summary API responses.
    
    Attributes:
        id: Summary ID
        video_url: YouTube video URL
        video_title: Video title
        video_thumbnail_url: Video thumbnail URL
        summary_text: Generated summary text
        summary_type: Type of summary
        summary_length: Length of summary
        transcript_language: Language of the transcript
        is_starred: Whether the summary is starred
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    id: str
    video_url: str
    video_title: Optional[str] = None
    video_thumbnail_url: Optional[str] = None
    summary_text: str
    summary_type: str
    summary_length: str
    transcript_language: Optional[str] = None
    is_starred: Optional[bool] = False
    created_at: datetime
    updated_at: datetime
    from_cache: Optional[bool] = False

class PaginatedSummaryResponse(BaseModel):
    """
    Model for paginated summary responses.
    
    Attributes:
        summaries: List of summaries
        pagination: Pagination metadata
    """
    summaries: List[SummaryResponse]
    pagination: Dict[str, Any]

class VideoQAResponse(BaseModel):
    """
    Model for video Q&A responses.
    
    Attributes:
        video_id: YouTube video ID
        video_title: Video title
        video_thumbnail_url: Video thumbnail URL
        history: Conversation history
        has_transcript: Whether the video has a transcript
        token_count: Number of tokens in the response
        transcript_token_count: Number of tokens in the transcript
    """
    video_id: str
    video_title: Optional[str] = None
    video_thumbnail_url: Optional[str] = None
    history: List[ChatMessage]
    has_transcript: bool
    token_count: Optional[int] = None
    transcript_token_count: Optional[int] = None

class VideoValidationResponse(BaseModel):
    """
    Model for video validation responses.
    
    Attributes:
        valid: Whether the URL is valid
        has_transcript: Whether the video has a transcript
        title: Video title
        thumbnail: Video thumbnail URL
        transcript_language: Language of the transcript
        message: Validation message
    """
    valid: bool
    has_transcript: bool
    title: Optional[str] = None
    thumbnail: Optional[str] = None
    transcript_language: Optional[str] = None
    message: str

class CacheStatsResponse(BaseModel):
    """
    Model for cache statistics responses.
    
    Attributes:
        status: Cache status
        used_memory_human: Human-readable used memory
        maxmemory_human: Human-readable maximum memory
        memory_percent: Memory usage percentage
        total_keys: Total number of keys in cache
        transcript_keys: Number of transcript keys
        video_info_keys: Number of video info keys
        languages_keys: Number of languages keys
        summary_keys: Number of summary keys
        task_keys: Number of task keys
        last_cleanup: Timestamp of last cleanup
    """
    status: str
    used_memory_human: str
    maxmemory_human: str
    memory_percent: str
    total_keys: int
    transcript_keys: int
    video_info_keys: int
    languages_keys: int
    summary_keys: int
    task_keys: int
    last_cleanup: str

class MemoryUsageResponse(BaseModel):
    """
    Model for memory usage responses.
    
    Attributes:
        timestamp: Timestamp of measurement
        memory_mb: Memory usage in MB
        percent: Memory usage percentage
        system_total_mb: Total system memory in MB
        system_available_mb: Available system memory in MB
        system_percent: System memory usage percentage
    """
    timestamp: float
    memory_mb: float
    memory_bytes: int
    percent: float
    system_total_mb: float
    system_available_mb: float
    system_percent: float

class HealthCheckResponse(BaseModel):
    """
    Model for health check responses.
    
    Attributes:
        status: Service status
        version: API version
        timestamp: Timestamp of health check
        uptime: Service uptime in seconds
        memory_usage: Memory usage information
        database_status: Database connection status
        cache_status: Cache connection status
        degraded_mode: Whether the service is in degraded mode
    """
    status: str
    version: str
    timestamp: datetime
    uptime: float
    memory_usage: MemoryUsageResponse
    database_status: str
    cache_status: str
    degraded_mode: bool

class ErrorResponse(BaseModel):
    """
    Model for error responses.
    
    Attributes:
        detail: Error message
        error_code: Error code
        timestamp: Timestamp of error
    """
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class CacheInvalidationResponse(BaseModel):
    """
    Model for cache invalidation responses.
    
    Attributes:
        success: Whether the invalidation was successful
        video_id: YouTube video ID
        invalidated_types: Cache types that were invalidated
        message: Invalidation message
    """
    success: bool
    video_id: str
    invalidated_types: List[str]
    message: str
