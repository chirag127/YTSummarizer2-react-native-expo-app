"""
API endpoints for video Q&A in the YouTube Summarizer API.

This module provides endpoints for asking questions about YouTube videos
and getting AI-generated answers based on the video transcript.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, Header, Request, Path, Query
from fastapi.responses import JSONResponse, StreamingResponse

from ...core.config import get_settings
from ...core.middleware import is_degraded_mode
from ...db import mongodb
from ...models.request_models import VideoQARequest, YouTubeURL, ChatMessage, ChatMessageRole
from ...models.response_models import VideoQAResponse
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
    "/video-qa/{video_id}",
    response_model=VideoQAResponse,
    summary="Ask a question about a video",
    description="Ask a question about a YouTube video and get an AI-generated answer based on the video transcript."
)
@throttled("video_qa")
async def video_qa(
    video_id: str = Path(..., description="YouTube video ID"),
    qa_request: VideoQARequest = None,
    request: Request = None,
    db = Depends(get_database),
    x_user_api_key: Optional[str] = Header(None)
):
    """
    Ask a question about a YouTube video.
    
    Args:
        video_id: YouTube video ID
        qa_request: Question and optional conversation history
        request: Request object
        db: MongoDB database
        x_user_api_key: Optional user-provided API key
    
    Returns:
        Answer and updated conversation history
    
    Raises:
        HTTPException: If video not found, no transcript is available, or an error occurs
    """
    # Check if in degraded mode
    if is_degraded_mode(request):
        raise HTTPException(
            status_code=503,
            detail="Service is in degraded mode. Video Q&A is temporarily unavailable. Please try again later."
        )
    
    # Get video info from cache or fetch it
    cached_video_info = await cache_service.get_cached_video_info(video_id)
    
    if cached_video_info:
        video_info = cached_video_info
    else:
        # Construct URL from video ID
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Fetch video info
        video_info = await youtube_service.extract_video_info(url)
    
    # Check if transcript is available
    if not video_info.get('transcript'):
        raise HTTPException(
            status_code=400,
            detail="No transcript available for this video. Cannot answer questions."
        )
    
    # Get conversation history or initialize new one
    history = qa_request.history or []
    
    # Create Gemini client with user API key if provided
    gemini_client = gemini_service.GeminiClient(api_key=x_user_api_key)
    
    try:
        # Generate answer
        answer = await gemini_client.generate_qa_response(
            transcript=video_info.get('transcript', ""),
            question=qa_request.question,
            history=[{
                "role": msg.role,
                "content": msg.content
            } for msg in history]
        )
        
        # Add question and answer to history
        now = datetime.now(timezone.utc)
        
        # Add user question
        history.append(ChatMessage(
            role=ChatMessageRole.USER,
            content=qa_request.question,
            timestamp=now
        ))
        
        # Add model response
        history.append(ChatMessage(
            role=ChatMessageRole.MODEL,
            content=answer,
            timestamp=now
        ))
        
        # Count tokens in transcript and history
        transcript_token_count = gemini_service.count_tokens(video_info.get('transcript', ""))
        history_token_count = sum(gemini_service.count_tokens(msg.content) for msg in history)
        
        # Return response
        return VideoQAResponse(
            video_id=video_id,
            video_title=video_info.get('title'),
            video_thumbnail_url=video_info.get('thumbnail'),
            history=history,
            has_transcript=True,
            token_count=history_token_count,
            transcript_token_count=transcript_token_count
        )
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        
        # If there's an error with the user's API key, return a specific error
        if x_user_api_key:
            raise HTTPException(
                status_code=400,
                detail="Failed to generate answer with your API key. Please check if your API key is valid and has sufficient quota."
            )
        
        # Otherwise, return a generic error
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )

@router.post(
    "/video-qa/{video_id}/stream",
    summary="Ask a question about a video (streaming)",
    description="Ask a question about a YouTube video and get a streaming AI-generated answer."
)
@throttled("video_qa")
async def video_qa_stream(
    video_id: str = Path(..., description="YouTube video ID"),
    qa_request: VideoQARequest = None,
    request: Request = None,
    db = Depends(get_database),
    x_user_api_key: Optional[str] = Header(None)
):
    """
    Ask a question about a YouTube video and get a streaming response.
    
    Args:
        video_id: YouTube video ID
        qa_request: Question and optional conversation history
        request: Request object
        db: MongoDB database
        x_user_api_key: Optional user-provided API key
    
    Returns:
        Streaming response with the answer
    
    Raises:
        HTTPException: If video not found, no transcript is available, or an error occurs
    """
    # Check if in degraded mode
    if is_degraded_mode(request):
        raise HTTPException(
            status_code=503,
            detail="Service is in degraded mode. Video Q&A is temporarily unavailable. Please try again later."
        )
    
    # Get video info from cache or fetch it
    cached_video_info = await cache_service.get_cached_video_info(video_id)
    
    if cached_video_info:
        video_info = cached_video_info
    else:
        # Construct URL from video ID
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Fetch video info
        video_info = await youtube_service.extract_video_info(url)
    
    # Check if transcript is available
    if not video_info.get('transcript'):
        raise HTTPException(
            status_code=400,
            detail="No transcript available for this video. Cannot answer questions."
        )
    
    # Get conversation history or initialize new one
    history = qa_request.history or []
    
    # Create Gemini client with user API key if provided
    gemini_client = gemini_service.GeminiClient(api_key=x_user_api_key)
    
    # Define streaming response generator
    async def generate_stream():
        try:
            # Generate streaming response
            async for chunk in gemini_client.generate_streaming_response(
                transcript=video_info.get('transcript', ""),
                question=qa_request.question,
                history=[{
                    "role": msg.role,
                    "content": msg.content
                } for msg in history]
            ):
                yield f"data: {chunk}\n\n"
            
            # End the stream
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error generating streaming answer: {e}")
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"
    
    # Return streaming response
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )

@router.get(
    "/video-qa/{video_id}/history",
    summary="Get conversation history",
    description="Get the conversation history for a video."
)
async def get_conversation_history(
    video_id: str = Path(..., description="YouTube video ID"),
    db = Depends(get_database)
):
    """
    Get the conversation history for a video.
    
    Args:
        video_id: YouTube video ID
        db: MongoDB database
    
    Returns:
        Conversation history
    """
    # Get conversation history from database
    history = await mongodb.execute_find_one(
        collection="video_chats",
        query={"videoId": video_id}
    )
    
    if not history:
        return {"videoId": video_id, "history": []}
    
    return history
