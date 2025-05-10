"""
YouTube service for the YouTube Summarizer API.

This module provides optimized YouTube video processing functionality,
including transcript extraction and video metadata retrieval.
"""

import os
import re
import logging
import asyncio
import random
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse, parse_qs
import yt_dlp
import requests
import aiohttp
from datetime import datetime

from ..core.config import get_settings
from ..core.circuit_breaker import async_circuit_breaker
from ..services import cache_service

# Configure logging
logger = logging.getLogger(__name__)

# YouTube URL regex pattern
YOUTUBE_URL_PATTERN = r'^(https?://)?(www\.|m\.)?(youtube\.com|youtu\.be)/.+$'

# Semaphore for limiting concurrent YouTube API requests
youtube_semaphore = asyncio.Semaphore(5)  # Default to 5, will be updated from config

def is_valid_youtube_url(url: str) -> bool:
    """
    Validate if the URL is a YouTube URL.
    
    Args:
        url: URL to validate
    
    Returns:
        True if valid YouTube URL, False otherwise
    """
    return bool(re.match(YOUTUBE_URL_PATTERN, str(url)))

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL.
    
    Args:
        url: YouTube URL
    
    Returns:
        Video ID or None if extraction fails
    """
    parsed_url = urlparse(url)
    
    if parsed_url.netloc == 'youtu.be':
        # Handle youtu.be URLs with query parameters
        # Extract only the path without query parameters
        video_id = parsed_url.path.lstrip('/')
        # Split at any potential query parameter
        video_id = video_id.split('?')[0]
        return video_id
    
    elif parsed_url.netloc in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                return query_params['v'][0]
        
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        
        elif parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
        
        elif parsed_url.path.startswith('/live/'):
            # Handle /live/ format URLs
            # Extract the video ID from the path
            video_id = parsed_url.path.split('/')[2]
            # Remove any query parameters
            video_id = video_id.split('?')[0]
            return video_id
    
    # If we get here, we can't extract the ID
    return None

def get_optimized_yt_dlp_options() -> Dict[str, Any]:
    """
    Get optimized yt-dlp options for resource-constrained environments.
    
    Returns:
        Dictionary of yt-dlp options
    """
    return {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'no_check_certificate': True,  # Skip certificate validation
        'socket_timeout': 30,          # 30 second socket timeout
        'no_playlist': True,           # Don't process playlists
        'extract_flat': True,          # Only extract basic metadata
        'force_generic_extractor': False,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'en-US', 'en-GB'],  # Prioritize English subtitles
        'retries': 3,                  # Number of retries for failed downloads
        'fragment_retries': 3,         # Number of retries for failed fragments
        'skip_unavailable_fragments': True,  # Skip unavailable fragments
        'extractor_retries': 3,        # Number of retries for failed extractions
    }

def get_minimal_yt_dlp_options() -> Dict[str, Any]:
    """
    Get minimal yt-dlp options for faster metadata retrieval.
    
    Returns:
        Dictionary of minimal yt-dlp options
    """
    return {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'no_check_certificate': True,
        'socket_timeout': 30,
        'no_playlist': True,
        'extract_flat': True,
        'force_generic_extractor': False,
        'writesubtitles': False,
        'writeautomaticsub': False,
    }

async def extract_video_info(url: str) -> Dict[str, Any]:
    """
    Extract video information using yt-dlp with caching.
    
    This is an optimized version that:
    1. Prioritizes cache usage
    2. Uses efficient yt-dlp options
    3. Properly handles async operations
    4. Implements resource constraints
    
    Args:
        url: YouTube URL
    
    Returns:
        Dictionary containing video information
    """
    # Extract video ID from URL
    video_id = extract_video_id(url)
    if not video_id:
        logger.error(f"Could not extract video ID from URL: {url}")
        return {
            'title': 'Title Unavailable',
            'thumbnail': None,
            'transcript': None,
            'error': "Could not extract video ID from URL"
        }
    
    # Check if video info is cached
    cached_video_info = await cache_service.get_cached_video_info(video_id)
    if cached_video_info:
        logger.info(f"Using cached video info for video ID: {video_id}")
        return cached_video_info
    
    # Check if transcript is cached
    cached_transcript = await cache_service.get_cached_transcript(video_id)
    if cached_transcript:
        logger.info(f"Using cached transcript for video ID: {video_id}")
        
        # We still need to fetch basic video info if not in cache
        try:
            # Use minimal yt-dlp options for faster metadata retrieval
            async with youtube_semaphore:
                # Run yt-dlp in a separate thread to avoid blocking
                info = await asyncio.to_thread(
                    _extract_video_metadata,
                    url,
                    get_minimal_yt_dlp_options()
                )
                
                video_info = {
                    'title': info.get('title', 'Title Unavailable'),
                    'thumbnail': info.get('thumbnail', None),
                    'transcript': cached_transcript.get('transcript'),
                    'transcript_language': cached_transcript.get('language'),
                    'video_id': video_id
                }
                
                # Cache the combined video info
                await cache_service.cache_video_info(video_id, video_info)
                return video_info
        except Exception as e:
            logger.error(f"Error fetching basic video info: {e}")
            # If we can't fetch basic info, at least return the transcript
            return {
                'title': 'Title Unavailable',
                'thumbnail': None,
                'transcript': cached_transcript.get('transcript'),
                'transcript_language': cached_transcript.get('language'),
                'video_id': video_id
            }
    
    # If not cached, proceed with full extraction
    try:
        async with youtube_semaphore:
            # Run yt-dlp in a separate thread to avoid blocking
            info = await asyncio.to_thread(
                _extract_video_metadata,
                url,
                get_optimized_yt_dlp_options()
            )
            
            # Extract relevant information
            video_info = {
                'title': info.get('title', 'Title Unavailable'),
                'thumbnail': info.get('thumbnail', None),
                'transcript': None,
                'transcript_language': None,
                'video_id': video_id
            }
            
            # Extract transcript
            transcript_data = await _extract_transcript(info, video_id)
            if transcript_data:
                video_info['transcript'] = transcript_data.get('transcript')
                video_info['transcript_language'] = transcript_data.get('language')
                
                # Cache the transcript separately
                await cache_service.cache_transcript(video_id, {
                    'transcript': transcript_data.get('transcript'),
                    'language': transcript_data.get('language')
                })
            
            # If we still don't have a transcript, try using the video description
            if not video_info.get('transcript') and info.get('description'):
                description = info.get('description', '')
                if len(description) > 200:  # Only use description if it's substantial
                    video_info['transcript'] = f"Video Description: {description}"
                    video_info['transcript_language'] = info.get('language') or 'unknown'
                    video_info['is_description_only'] = True
                    
                    # Cache the description as transcript
                    await cache_service.cache_transcript(video_id, {
                        'transcript': f"Video Description: {description}",
                        'language': info.get('language') or 'unknown'
                    })
            
            # Cache the full video info
            if video_info.get('transcript'):
                await cache_service.cache_video_info(video_id, video_info)
                
                # Also cache available languages if we have them
                if info.get('subtitles') or info.get('automatic_captions'):
                    languages = {
                        'subtitles': list(info.get('subtitles', {}).keys()),
                        'automatic_captions': list(info.get('automatic_captions', {}).keys())
                    }
                    await cache_service.cache_available_languages(video_id, languages)
            
            return video_info
    except Exception as e:
        logger.error(f"Error extracting video info: {e}")
        return {
            'title': 'Title Unavailable',
            'thumbnail': None,
            'transcript': None,
            'error': str(e)
        }

def _extract_video_metadata(url: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract video metadata using yt-dlp.
    
    This function runs in a separate thread to avoid blocking the event loop.
    
    Args:
        url: YouTube URL
        options: yt-dlp options
    
    Returns:
        Dictionary containing video metadata
    """
    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=False)

async def _extract_transcript(info: Dict[str, Any], video_id: str) -> Optional[Dict[str, Any]]:
    """
    Extract transcript from video info.
    
    Args:
        info: Video info from yt-dlp
        video_id: YouTube video ID
    
    Returns:
        Dictionary containing transcript text and language, or None if not available
    """
    transcript_text = ""
    transcript_lang = None
    
    # Try to get manual subtitles first
    if info.get('subtitles'):
        transcript_data = await _extract_subtitles(info.get('subtitles', {}), 'en')
        if transcript_data:
            return transcript_data
        
        # If no English subtitles, try any other available language
        available_langs = list(info.get('subtitles', {}).keys())
        logger.info(f"Available subtitle languages: {available_langs}")
        
        for lang in available_langs:
            if lang == 'en':  # Already tried English
                continue
            
            transcript_data = await _extract_subtitles(info.get('subtitles', {}), lang)
            if transcript_data:
                return transcript_data
    
    # If no manual subtitles, try auto-generated captions
    if info.get('automatic_captions'):
        transcript_data = await _extract_subtitles(info.get('automatic_captions', {}), 'en')
        if transcript_data:
            return transcript_data
        
        # If no English auto-captions, try any other available language
        available_langs = list(info.get('automatic_captions', {}).keys())
        logger.info(f"Available auto-caption languages: {available_langs}")
        
        for lang in available_langs:
            if lang == 'en':  # Already tried English
                continue
            
            transcript_data = await _extract_subtitles(info.get('automatic_captions', {}), lang)
            if transcript_data:
                return transcript_data
    
    # If we still don't have a transcript, try using the YouTube transcript API as a fallback
    if video_id:
        transcript_data = await _extract_transcript_from_api(video_id)
        if transcript_data:
            return transcript_data
    
    return None

async def _extract_subtitles(
    subtitles: Dict[str, List[Dict[str, Any]]],
    lang: str
) -> Optional[Dict[str, Any]]:
    """
    Extract subtitles for a specific language.
    
    Args:
        subtitles: Dictionary of subtitles from yt-dlp
        lang: Language code
    
    Returns:
        Dictionary containing transcript text and language, or None if not available
    """
    subs = subtitles.get(lang, [])
    if not subs:
        return None
    
    for format_dict in subs:
        if format_dict.get('ext') in ['vtt', 'srt']:
            try:
                # Download the subtitle file
                sub_url = format_dict.get('url')
                async with aiohttp.ClientSession() as session:
                    async with session.get(sub_url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            transcript_text = _parse_subtitle_content(content)
                            
                            if transcript_text:
                                logger.info(f"Using subtitles in language: {lang}")
                                return {
                                    'transcript': transcript_text,
                                    'language': lang
                                }
            except Exception as e:
                logger.error(f"Error downloading {lang} subtitles: {e}")
    
    return None

def _parse_subtitle_content(content: str) -> str:
    """
    Parse subtitle content (VTT/SRT) into plain text.
    
    Args:
        content: Subtitle content
    
    Returns:
        Plain text transcript
    """
    transcript_text = ""
    
    # Split into lines
    lines = content.split('\n')
    
    for line in lines:
        # Skip timing lines, empty lines, and metadata
        if (re.match(r'^\d+:\d+:\d+', line) or
            re.match(r'^\d+$', line) or
            line.strip() == '' or
            line.startswith('WEBVTT')):
            continue
        
        # Remove HTML tags
        clean_line = re.sub(r'<[^>]+>', '', line)
        
        if clean_line.strip():
            transcript_text += clean_line.strip() + ' '
    
    return transcript_text.strip()

async def _extract_transcript_from_api(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Extract transcript using YouTube's transcript API.
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        Dictionary containing transcript text and language, or None if not available
    """
    # First try English
    try:
        transcript_url = f"https://www.youtube.com/api/timedtext?lang=en&v={video_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(transcript_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    if content:
                        # Parse the XML response
                        transcript_text = _parse_xml_transcript(content)
                        if transcript_text:
                            return {
                                'transcript': transcript_text,
                                'language': 'en'
                            }
    except Exception as e:
        logger.error(f"Error using YouTube transcript API (English): {e}")
    
    # If English transcript not available, try to get a list of available languages
    try:
        lang_list_url = f"https://www.youtube.com/api/timedtext?type=list&v={video_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(lang_list_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    if content:
                        # Extract language codes from XML
                        lang_codes = re.findall(r'lang_code="([^"]+)"', content)
                        logger.info(f"Available transcript languages: {lang_codes}")
                        
                        # Try each language until we find one that works
                        for lang in lang_codes:
                            if lang == 'en':  # Already tried English
                                continue
                            
                            try:
                                transcript_url = f"https://www.youtube.com/api/timedtext?lang={lang}&v={video_id}"
                                async with session.get(transcript_url, timeout=10) as response:
                                    if response.status == 200:
                                        content = await response.text()
                                        if content:
                                            # Parse the XML response
                                            transcript_text = _parse_xml_transcript(content)
                                            if transcript_text:
                                                logger.info(f"Using transcript in language: {lang}")
                                                return {
                                                    'transcript': transcript_text,
                                                    'language': lang
                                                }
                            except Exception as e:
                                logger.error(f"Error using YouTube transcript API for language {lang}: {e}")
    except Exception as e:
        logger.error(f"Error getting available transcript languages: {e}")
    
    return None

def _parse_xml_transcript(content: str) -> str:
    """
    Parse XML transcript content.
    
    Args:
        content: XML transcript content
    
    Returns:
        Plain text transcript
    """
    transcript_text = ""
    
    # Extract text from XML
    text_matches = re.findall(r'<text[^>]*>(.*?)</text>', content)
    for text in text_matches:
        # Decode HTML entities
        decoded_text = (text.replace('&amp;', '&')
                            .replace('&lt;', '<')
                            .replace('&gt;', '>')
                            .replace('&quot;', '"')
                            .replace('&#39;', "'"))
        transcript_text += decoded_text + ' '
    
    return transcript_text.strip()

async def process_transcript_in_chunks(
    transcript: str,
    max_tokens_per_chunk: int = 1000,
    delay_between_batches: int = 5
) -> List[str]:
    """
    Process transcript in chunks to avoid memory issues.
    
    Args:
        transcript: Full transcript text
        max_tokens_per_chunk: Maximum tokens per chunk
        delay_between_batches: Delay between processing batches in seconds
    
    Returns:
        List of processed transcript chunks
    """
    if not transcript:
        return []
    
    # Split transcript into sentences
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    
    # Group sentences into chunks of approximately max_tokens_per_chunk
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    for sentence in sentences:
        sentence_token_count = len(sentence) // 4 + 1  # Add 1 to account for potential underestimation
        
        if current_token_count + sentence_token_count > max_tokens_per_chunk and current_chunk:
            # Current chunk is full, add it to chunks and start a new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_token_count = sentence_token_count
            
            # Add delay between batches to avoid memory spikes
            if len(chunks) % 5 == 0:  # Every 5 chunks
                await asyncio.sleep(delay_between_batches)
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

async def init_youtube_service() -> None:
    """Initialize the YouTube service."""
    global youtube_semaphore
    
    # Load settings
    settings = get_settings()
    max_connections = settings.resource_limits.YOUTUBE_API_MAX_CONNECTIONS
    
    # Update semaphore with configured value
    youtube_semaphore = asyncio.Semaphore(max_connections)
    
    logger.info(f"YouTube service initialized with max {max_connections} concurrent connections")
