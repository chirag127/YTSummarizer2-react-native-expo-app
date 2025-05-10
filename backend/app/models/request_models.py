"""
Request models for the YouTube Summarizer API.

This module defines Pydantic models for request validation with
optimized constraints for resource-constrained environments.
"""

import re
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator, root_validator, HttpUrl

# Maximum URL length: 2048 characters
MAX_URL_LENGTH = 2048

# YouTube URL regex pattern
YOUTUBE_URL_PATTERN = r'^(https?://)?(www\.|m\.)?(youtube\.com|youtu\.be)/.+$'

class SummaryType(str, Enum):
    """Types of summaries that can be generated."""
    BRIEF = "Brief"
    DETAILED = "Detailed"
    KEY_POINT = "Key Point"
    CHAPTERS = "Chapters"

class SummaryLength(str, Enum):
    """Length options for summaries."""
    SHORT = "Short"
    MEDIUM = "Medium"
    LONG = "Long"

class ChatMessageRole(str, Enum):
    """Roles for chat messages."""
    USER = "user"
    MODEL = "model"
    SYSTEM = "system"

class YouTubeURL(BaseModel):
    """
    Model for YouTube URL validation and summary generation parameters.
    
    Attributes:
        url: YouTube video URL
        summary_type: Type of summary to generate
        summary_length: Length of summary to generate
        force_regenerate: Whether to force regeneration of summary
    """
    url: str = Field(..., max_length=MAX_URL_LENGTH)
    summary_type: SummaryType = Field(default=SummaryType.BRIEF)
    summary_length: SummaryLength = Field(default=SummaryLength.MEDIUM)
    force_regenerate: bool = Field(default=False)
    
    @validator('url')
    def validate_youtube_url(cls, v):
        """Validate that the URL is a YouTube URL."""
        if not re.match(YOUTUBE_URL_PATTERN, v):
            raise ValueError("Invalid YouTube URL")
        if len(v) > MAX_URL_LENGTH:
            raise ValueError(f"URL length exceeds maximum allowed ({MAX_URL_LENGTH} characters)")
        return v

class ChatMessage(BaseModel):
    """
    Model for chat messages.
    
    Attributes:
        role: Role of the message sender (user or model)
        content: Message content
        timestamp: Message timestamp
    """
    role: ChatMessageRole
    content: str = Field(..., max_length=10000)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('content')
    def validate_content(cls, v):
        """Validate that the content is not empty."""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v

class VideoQARequest(BaseModel):
    """
    Model for video Q&A requests.
    
    Attributes:
        question: User's question about the video
        history: Optional conversation history
    """
    question: str = Field(..., min_length=1, max_length=1000)
    history: Optional[List[ChatMessage]] = None
    
    @validator('question')
    def validate_question(cls, v):
        """Validate that the question is not empty."""
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v
    
    @root_validator
    def validate_history_length(cls, values):
        """Validate that the history is not too long."""
        history = values.get('history')
        if history and len(history) > 50:
            raise ValueError("Conversation history is too long (maximum 50 messages)")
        return values

class SummaryUpdate(BaseModel):
    """
    Model for updating summary parameters.
    
    Attributes:
        summary_type: New summary type
        summary_length: New summary length
    """
    summary_type: Optional[SummaryType] = None
    summary_length: Optional[SummaryLength] = None

class StarUpdate(BaseModel):
    """
    Model for updating starred status.
    
    Attributes:
        is_starred: Whether the summary is starred
    """
    is_starred: bool

class CacheInvalidationRequest(BaseModel):
    """
    Model for cache invalidation requests.
    
    Attributes:
        video_id: YouTube video ID to invalidate cache for
        invalidate_all: Whether to invalidate all caches for the video
        invalidate_types: Specific cache types to invalidate
    """
    video_id: str = Field(..., min_length=5, max_length=20)
    invalidate_all: bool = Field(default=False)
    invalidate_types: Optional[List[str]] = Field(
        default=None,
        description="Cache types to invalidate: video_info, transcript, summary"
    )
    
    @validator('video_id')
    def validate_video_id(cls, v):
        """Validate that the video ID is in the correct format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Invalid YouTube video ID format")
        return v
    
    @validator('invalidate_types')
    def validate_invalidate_types(cls, v):
        """Validate that the invalidate types are valid."""
        if v is None:
            return v
        valid_types = ["video_info", "transcript", "summary"]
        for t in v:
            if t not in valid_types:
                raise ValueError(f"Invalid cache type: {t}. Valid types are: {', '.join(valid_types)}")
        return v
