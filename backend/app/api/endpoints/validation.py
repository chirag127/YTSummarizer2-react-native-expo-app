"""
API endpoints for URL validation in the YouTube Summarizer API.

This module provides endpoints for validating YouTube URLs and
checking if they have available transcripts for summarization.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse

from ...core.config import get_settings
from ...models.request_models import YouTubeURL
from ...models.response_models import VideoValidationResponse
from ...services import youtube_service, cache_service
from ...utils.rate_limiter import throttled
from ...utils.request_throttler import throttled

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post(
    "/validate-url",
    response_model=VideoValidationResponse,
    summary="Validate YouTube URL",
    description="Validates a YouTube URL and checks if it has an available transcript for summarization."
)
@throttled("validate_url")
async def validate_url(
    youtube_url: YouTubeURL,
    request: Request,
    x_user_api_key: Optional[str] = Header(None)
):
    """
    Validate YouTube URL and extract basic information.
    
    Args:
        youtube_url: YouTube URL to validate
        request: Request object
        x_user_api_key: Optional user-provided API key
    
    Returns:
        Validation result with video information
    
    Raises:
        HTTPException: If URL is invalid or an error occurs
    """
    url = str(youtube_url.url)
    
    if not youtube_service.is_valid_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    try:
        # Extract video ID
        video_id = youtube_service.extract_video_id(url)
        if not video_id:
            return VideoValidationResponse(
                valid=True,
                has_transcript=False,
                message="Could not extract video ID from URL"
            )
        
        # Check cache first
        cached_video_info = await cache_service.get_cached_video_info(video_id)
        if cached_video_info:
            logger.info(f"Using cached video info for video ID: {video_id}")
            
            has_transcript = cached_video_info.get('transcript') is not None
            
            return VideoValidationResponse(
                valid=True,
                has_transcript=has_transcript,
                title=cached_video_info.get('title'),
                thumbnail=cached_video_info.get('thumbnail'),
                transcript_language=cached_video_info.get('transcript_language'),
                message="Valid YouTube URL" + (" with available transcript." if has_transcript else " without available transcript.")
            )
        
        # If not in cache, extract video info
        video_info = await youtube_service.extract_video_info(url)
        
        if not video_info.get('transcript'):
            return VideoValidationResponse(
                valid=True,
                has_transcript=False,
                title=video_info.get('title'),
                thumbnail=video_info.get('thumbnail'),
                message="Video found, but no transcript/captions available for summarization."
            )
        
        return VideoValidationResponse(
            valid=True,
            has_transcript=True,
            title=video_info.get('title'),
            thumbnail=video_info.get('thumbnail'),
            transcript_language=video_info.get('transcript_language'),
            message="Valid YouTube URL with available transcript."
        )
    except Exception as e:
        logger.error(f"Error validating URL: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

@router.get(
    "/cache/stats",
    summary="Get cache statistics",
    description="Get statistics about the cache usage."
)
async def get_cache_stats():
    """
    Get cache statistics.
    
    Returns:
        Cache statistics
    """
    stats = await cache_service.get_cache_stats()
    return stats

@router.post(
    "/cache/invalidate",
    summary="Invalidate cache",
    description="Invalidate cache for a specific video."
)
async def invalidate_cache(
    video_id: str,
    invalidate_all: bool = False,
    invalidate_types: Optional[str] = None
):
    """
    Invalidate cache for a video.
    
    Args:
        video_id: YouTube video ID
        invalidate_all: Whether to invalidate all caches for the video
        invalidate_types: Comma-separated list of cache types to invalidate
    
    Returns:
        Invalidation result
    """
    # Parse invalidate_types
    types_list = None
    if invalidate_types:
        types_list = [t.strip() for t in invalidate_types.split(",")]
    
    # Invalidate cache
    invalidated = await cache_service.invalidate_video_cache(
        video_id=video_id,
        invalidate_all=invalidate_all,
        invalidate_types=types_list
    )
    
    return {
        "success": True,
        "video_id": video_id,
        "invalidated_types": invalidated,
        "message": f"Cache invalidated for video {video_id}"
    }
