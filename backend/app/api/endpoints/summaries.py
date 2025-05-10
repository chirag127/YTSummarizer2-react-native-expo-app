"""
API endpoints for summaries in the YouTube Summarizer API.

This module provides endpoints for generating and retrieving summaries
of YouTube videos.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, Header, Request, Query, Path
from fastapi.responses import JSONResponse

from ...core.config import get_settings
from ...core.middleware import is_degraded_mode
from ...db import mongodb
from ...models.request_models import YouTubeURL, SummaryUpdate, StarUpdate
from ...models.response_models import SummaryResponse, PaginatedSummaryResponse
from ...services import youtube_service, gemini_service, cache_service
from ...utils.rate_limiter import throttled
from ...utils.request_throttler import throttled

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Helper function to get database
async def get_database():
    """Get MongoDB database instance."""
    return await mongodb.get_database()

@router.post(
    "/generate-summary",
    response_model=SummaryResponse,
    summary="Generate summary",
    description="Generate a summary for a YouTube video."
)
@throttled("ai_summarization")
async def create_summary(
    youtube_url: YouTubeURL,
    request: Request,
    db = Depends(get_database),
    x_user_api_key: Optional[str] = Header(None)
):
    """
    Generate summary for a YouTube video and store it.
    
    Args:
        youtube_url: YouTube URL and summary parameters
        request: Request object
        db: MongoDB database
        x_user_api_key: Optional user-provided API key
    
    Returns:
        Generated summary
    
    Raises:
        HTTPException: If URL is invalid, no transcript is available, or an error occurs
    """
    url = str(youtube_url.url)
    
    # Check if in degraded mode
    if is_degraded_mode(request) and not youtube_url.force_regenerate:
        logger.warning("Request for summary generation in degraded mode")
        
        # In degraded mode, only allow cached summaries
        video_id = youtube_service.extract_video_id(url)
        if video_id:
            # Check if summary exists in cache
            cached_summary = await cache_service.get_cached_summary(
                video_id,
                youtube_url.summary_type,
                youtube_url.summary_length
            )
            
            if cached_summary:
                logger.info(f"Using cached summary in degraded mode for video ID {video_id}")
                
                # Get video info for title and thumbnail
                video_info = await cache_service.get_cached_video_info(video_id)
                
                summary = {
                    "id": f"cache:{video_id}:{youtube_url.summary_type}:{youtube_url.summary_length}",
                    "video_url": url,
                    "video_title": video_info.get('title', 'Title Unavailable') if video_info else 'Title Unavailable',
                    "video_thumbnail_url": video_info.get('thumbnail') if video_info else None,
                    "summary_text": cached_summary['summary_text'],
                    "summary_type": youtube_url.summary_type,
                    "summary_length": youtube_url.summary_length,
                    "transcript_language": video_info.get('transcript_language') if video_info else None,
                    "is_starred": False,
                    "created_at": datetime.fromisoformat(cached_summary.get('generated_at', datetime.now(timezone.utc).isoformat())),
                    "updated_at": datetime.now(timezone.utc),
                    "from_cache": True
                }
                
                return SummaryResponse(**summary)
            
            # If not in cache, check database
            existing_summary = await mongodb.execute_find_one(
                collection="summaries",
                query={
                    "video_url": url,
                    "summary_type": youtube_url.summary_type,
                    "summary_length": youtube_url.summary_length
                }
            )
            
            if existing_summary:
                # Also cache this summary for faster future access
                await cache_service.cache_summary_result(
                    video_id,
                    youtube_url.summary_type,
                    youtube_url.summary_length,
                    existing_summary.get("summary_text", "")
                )
                
                return SummaryResponse(**existing_summary)
        
        # If we get here, we can't serve from cache or database
        raise HTTPException(
            status_code=503,
            detail="Service is in degraded mode. Only cached summaries are available. Please try again later."
        )
    
    if not youtube_service.is_valid_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Check if summary already exists with the same URL, type, and length
    # Skip this check if force_regenerate is True
    if not youtube_url.force_regenerate:
        # First check Redis cache for faster response
        video_id = youtube_service.extract_video_id(url)
        if video_id:
            cached_summary = await cache_service.get_cached_summary(
                video_id,
                youtube_url.summary_type,
                youtube_url.summary_length
            )
            
            if cached_summary and 'summary_text' in cached_summary:
                logger.info(f"Using cached summary for video ID {video_id}")
                
                # Get video info for title and thumbnail
                video_info = await cache_service.get_cached_video_info(video_id)
                
                summary = {
                    "id": f"cache:{video_id}:{youtube_url.summary_type}:{youtube_url.summary_length}",
                    "video_url": url,
                    "video_title": video_info.get('title', 'Title Unavailable') if video_info else 'Title Unavailable',
                    "video_thumbnail_url": video_info.get('thumbnail') if video_info else None,
                    "summary_text": cached_summary['summary_text'],
                    "summary_type": youtube_url.summary_type,
                    "summary_length": youtube_url.summary_length,
                    "transcript_language": video_info.get('transcript_language') if video_info else None,
                    "is_starred": False,
                    "created_at": datetime.fromisoformat(cached_summary.get('generated_at', datetime.now(timezone.utc).isoformat())),
                    "updated_at": datetime.now(timezone.utc),
                    "from_cache": True
                }
                
                return SummaryResponse(**summary)
        
        # If not in Redis cache, check database
        existing_summary = await mongodb.execute_find_one(
            collection="summaries",
            query={
                "video_url": url,
                "summary_type": youtube_url.summary_type,
                "summary_length": youtube_url.summary_length
            }
        )
        
        if existing_summary:
            # Also cache this summary for faster future access
            if video_id:
                await cache_service.cache_summary_result(
                    video_id,
                    youtube_url.summary_type,
                    youtube_url.summary_length,
                    existing_summary.get("summary_text", "")
                )
            
            return SummaryResponse(**existing_summary)
    
    # Extract video information
    video_info = await youtube_service.extract_video_info(url)
    
    if not video_info.get('transcript'):
        raise HTTPException(
            status_code=400,
            detail="No transcript/captions available for this video. Cannot generate summary."
        )
    
    # Generate summary
    try:
        # Create Gemini client with user API key if provided
        gemini_client = gemini_service.GeminiClient(api_key=x_user_api_key)
        
        summary_text = await gemini_client.generate_summary(
            video_info.get('transcript', "No transcript available"),
            youtube_url.summary_type,
            youtube_url.summary_length
        )
    except Exception as e:
        # If there's an error with the user's API key, log it and return a specific error
        if x_user_api_key:
            logger.error(f"Error generating summary with user API key: {e}")
            raise HTTPException(
                status_code=400,
                detail="Failed to generate summary with your API key. Please check if your API key is valid and has sufficient quota."
            )
        # If using the default API key, re-raise the exception
        raise
    
    # Create summary document
    now = datetime.now(timezone.utc)
    summary = {
        "video_url": url,
        "video_title": video_info.get('title', 'Title Unavailable'),
        "video_thumbnail_url": video_info.get('thumbnail'),
        "summary_text": summary_text,
        "summary_type": youtube_url.summary_type,
        "summary_length": youtube_url.summary_length,
        "transcript_language": video_info.get('transcript_language'),
        "is_starred": False,
        "created_at": now,
        "updated_at": now
    }
    
    # Insert into database
    result_id = await mongodb.execute_insert_one(
        collection="summaries",
        document=summary
    )
    
    if not result_id:
        raise HTTPException(
            status_code=500,
            detail="Failed to store summary in database"
        )
    
    # Also cache the summary for faster future access
    video_id = youtube_service.extract_video_id(url)
    if video_id:
        await cache_service.cache_summary_result(
            video_id,
            youtube_url.summary_type,
            youtube_url.summary_length,
            summary_text
        )
        logger.info(f"Cached summary for video ID {video_id}")
    
    # Return response
    summary["id"] = result_id
    return SummaryResponse(**summary)

@router.get(
    "/summaries",
    response_model=PaginatedSummaryResponse,
    summary="Get summaries",
    description="Get summaries with pagination."
)
async def get_summaries(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    video_url: Optional[str] = Query(None, description="Filter by video URL"),
    is_starred: Optional[bool] = Query(None, description="Filter by starred status"),
    db = Depends(get_database)
):
    """
    Get summaries with pagination.
    
    Args:
        page: Page number (1-based)
        limit: Items per page
        video_url: Filter by video URL
        is_starred: Filter by starred status
        db: MongoDB database
    
    Returns:
        Paginated list of summaries
    """
    # Ensure valid pagination parameters
    page = max(1, page)  # Minimum page is 1
    limit = min(max(1, limit), 100)  # Limit between 1 and 100
    skip = (page - 1) * limit
    
    # Build query filter
    query_filter = {}
    if video_url:
        query_filter["video_url"] = video_url
    if is_starred is not None:
        query_filter["is_starred"] = is_starred
    
    # Get total count for pagination info
    total_count = await mongodb.execute_count(
        collection="summaries",
        query=query_filter
    )
    
    # Get paginated summaries
    summaries = await mongodb.execute_find(
        collection="summaries",
        query=query_filter,
        sort=[("created_at", -1)],
        skip=skip,
        limit=limit
    )
    
    # Calculate pagination metadata
    total_pages = (total_count + limit - 1) // limit  # Ceiling division
    has_next = page < total_pages
    has_prev = page > 1
    
    # Return summaries with pagination metadata
    return {
        "summaries": [SummaryResponse(**summary) for summary in summaries],
        "pagination": {
            "page": page,
            "limit": limit,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev
        }
    }

@router.get(
    "/summaries/{summary_id}",
    response_model=SummaryResponse,
    summary="Get summary",
    description="Get a specific summary by ID."
)
async def get_summary(
    summary_id: str = Path(..., description="Summary ID"),
    db = Depends(get_database)
):
    """
    Get a specific summary by ID.
    
    Args:
        summary_id: Summary ID
        db: MongoDB database
    
    Returns:
        Summary
    
    Raises:
        HTTPException: If summary not found
    """
    summary = await mongodb.execute_find_one(
        collection="summaries",
        query={"id": summary_id}
    )
    
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    return SummaryResponse(**summary)

@router.put(
    "/summaries/{summary_id}",
    response_model=SummaryResponse,
    summary="Update summary",
    description="Update a summary with new parameters."
)
@throttled("ai_summarization")
async def update_summary(
    summary_id: str = Path(..., description="Summary ID"),
    update_data: SummaryUpdate = None,
    db = Depends(get_database),
    x_user_api_key: Optional[str] = Header(None)
):
    """
    Create a new summary with updated parameters.
    
    Args:
        summary_id: Summary ID
        update_data: Update data
        db: MongoDB database
        x_user_api_key: Optional user-provided API key
    
    Returns:
        Updated summary
    
    Raises:
        HTTPException: If summary not found or an error occurs
    """
    # Find the existing summary
    existing_summary = await mongodb.execute_find_one(
        collection="summaries",
        query={"id": summary_id}
    )
    
    if not existing_summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    # Determine new parameters
    summary_type = update_data.summary_type or existing_summary["summary_type"]
    summary_length = update_data.summary_length or existing_summary["summary_length"]
    
    # Check if we're actually changing the type or length
    if summary_type == existing_summary["summary_type"] and summary_length == existing_summary["summary_length"]:
        # No change, just return the existing summary
        return SummaryResponse(**existing_summary)
    
    # Check if a summary with these parameters already exists
    existing_with_params = await mongodb.execute_find_one(
        collection="summaries",
        query={
            "video_url": existing_summary["video_url"],
            "summary_type": summary_type,
            "summary_length": summary_length
        }
    )
    
    if existing_with_params:
        # Return the existing summary with these parameters
        return SummaryResponse(**existing_with_params)
    
    # Extract video information again
    video_info = await youtube_service.extract_video_info(existing_summary["video_url"])
    
    if not video_info.get('transcript'):
        raise HTTPException(
            status_code=400,
            detail="No transcript/captions available for this video. Cannot regenerate summary."
        )
    
    # Generate new summary
    try:
        # Create Gemini client with user API key if provided
        gemini_client = gemini_service.GeminiClient(api_key=x_user_api_key)
        
        summary_text = await gemini_client.generate_summary(
            video_info.get('transcript', "No transcript available"),
            summary_type,
            summary_length
        )
    except Exception as e:
        # If there's an error with the user's API key, log it and return a specific error
        if x_user_api_key:
            logger.error(f"Error generating summary with user API key: {e}")
            raise HTTPException(
                status_code=400,
                detail="Failed to generate summary with your API key. Please check if your API key is valid and has sufficient quota."
            )
        # If using the default API key, re-raise the exception
        raise
    
    # Create a new summary document
    now = datetime.now(timezone.utc)
    new_summary = {
        "video_url": existing_summary["video_url"],
        "video_title": existing_summary["video_title"],
        "video_thumbnail_url": existing_summary["video_thumbnail_url"],
        "summary_text": summary_text,
        "summary_type": summary_type,
        "summary_length": summary_length,
        "transcript_language": existing_summary.get("transcript_language"),
        "is_starred": existing_summary.get("is_starred", False),
        "created_at": now,
        "updated_at": now
    }
    
    # Insert into database
    result_id = await mongodb.execute_insert_one(
        collection="summaries",
        document=new_summary
    )
    
    if not result_id:
        raise HTTPException(
            status_code=500,
            detail="Failed to store summary in database"
        )
    
    # Also cache the summary for faster future access
    video_id = youtube_service.extract_video_id(existing_summary["video_url"])
    if video_id:
        await cache_service.cache_summary_result(
            video_id,
            summary_type,
            summary_length,
            summary_text
        )
    
    # Return response
    new_summary["id"] = result_id
    return SummaryResponse(**new_summary)

@router.patch(
    "/summaries/{summary_id}/star",
    response_model=SummaryResponse,
    summary="Update star status",
    description="Update the starred status of a summary."
)
async def update_star_status(
    summary_id: str = Path(..., description="Summary ID"),
    star_update: StarUpdate = None,
    db = Depends(get_database)
):
    """
    Update the starred status of a summary.
    
    Args:
        summary_id: Summary ID
        star_update: Star update data
        db: MongoDB database
    
    Returns:
        Updated summary
    
    Raises:
        HTTPException: If summary not found or update fails
    """
    # Find the existing summary
    existing_summary = await mongodb.execute_find_one(
        collection="summaries",
        query={"id": summary_id}
    )
    
    if not existing_summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    # Update the starred status
    now = datetime.now(timezone.utc)
    result = await mongodb.execute_update_one(
        collection="summaries",
        query={"id": summary_id},
        update={
            "$set": {
                "is_starred": star_update.is_starred,
                "updated_at": now
            }
        }
    )
    
    if not result:
        raise HTTPException(
            status_code=500,
            detail="Failed to update summary"
        )
    
    # Get the updated summary
    updated_summary = await mongodb.execute_find_one(
        collection="summaries",
        query={"id": summary_id}
    )
    
    return SummaryResponse(**updated_summary)

@router.delete(
    "/summaries/{summary_id}",
    summary="Delete summary",
    description="Delete a summary."
)
async def delete_summary(
    summary_id: str = Path(..., description="Summary ID"),
    db = Depends(get_database)
):
    """
    Delete a summary.
    
    Args:
        summary_id: Summary ID
        db: MongoDB database
    
    Returns:
        Deletion result
    
    Raises:
        HTTPException: If summary not found or deletion fails
    """
    # Find the existing summary
    existing_summary = await mongodb.execute_find_one(
        collection="summaries",
        query={"id": summary_id}
    )
    
    if not existing_summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    # Delete the summary
    result = await mongodb.execute_delete_one(
        collection="summaries",
        query={"id": summary_id}
    )
    
    if not result:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete summary"
        )
    
    # Also invalidate cache if possible
    video_id = youtube_service.extract_video_id(existing_summary["video_url"])
    if video_id:
        await cache_service.invalidate_video_cache(
            video_id=video_id,
            invalidate_types=["summary"]
        )
    
    return {"message": "Summary deleted successfully"}
