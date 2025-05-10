"""
Gemini API service for the YouTube Summarizer API.

This module provides optimized Gemini API integration for generating
summaries and answering questions about YouTube videos.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, AsyncGenerator
import google.generativeai as genai
from google.generativeai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..core.config import get_settings
from ..core.circuit_breaker import async_circuit_breaker
from ..models.request_models import ChatMessageRole

# Configure logging
logger = logging.getLogger(__name__)

# Semaphore for limiting concurrent Gemini API requests
gemini_semaphore = asyncio.Semaphore(3)  # Default to 3, will be updated from config

# Token limits
MAX_TOTAL_TOKENS = 1048576  # Maximum tokens for the entire input context
MAX_TRANSCRIPT_TOKENS = 800000  # Maximum tokens for the transcript
MAX_HISTORY_TOKENS = 150000  # Maximum tokens for conversation history
MAX_QUESTION_TOKENS = 2000  # Maximum tokens for the current question
MAX_OUTPUT_TOKENS = 1000  # Maximum tokens for the model's response
RESERVE_TOKENS = 65536  # Reserve tokens for the model's response

# Fallback token counting for when tiktoken is not available
def count_tokens_fallback(text: str) -> int:
    """
    Estimate token count using a simple heuristic.
    This is a fallback method when tiktoken is not available.
    
    Args:
        text: The text to count tokens for
    
    Returns:
        Estimated token count
    """
    # Simple heuristic: ~4 characters per token on average
    return len(text) // 4

# Try to use tiktoken for more accurate token counting
try:
    import tiktoken
    
    # Initialize the tokenizer
    # Using cl100k_base which is close to what Gemini models use
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: The text to count tokens for
        
        Returns:
            Token count
        """
        if not text:
            return 0
        return len(tokenizer.encode(text))
except ImportError:
    logger.warning("tiktoken not installed. Using fallback token counting method.")
    count_tokens = count_tokens_fallback

def truncate_transcript(transcript: str, max_tokens: int = MAX_TRANSCRIPT_TOKENS) -> str:
    """
    Truncate transcript to fit within token limit.
    
    Args:
        transcript: The full transcript text
        max_tokens: Maximum allowed tokens
    
    Returns:
        Truncated transcript
    """
    if not transcript:
        return ""
    
    # Count tokens in the transcript
    token_count = count_tokens(transcript)
    
    # If within limits, return as is
    if token_count <= max_tokens:
        return transcript
    
    # Otherwise, truncate
    logger.info(f"Truncating transcript from {token_count} tokens to {max_tokens} tokens")
    
    # Simple truncation approach - split into sentences and keep adding until limit
    sentences = transcript.split('. ')
    truncated = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence + '. ')
        if current_tokens + sentence_tokens <= max_tokens:
            truncated.append(sentence)
            current_tokens += sentence_tokens
        else:
            break
    
    result = '. '.join(truncated)
    if not result.endswith('.'):
        result += '.'
    
    logger.info(f"Truncated transcript to {count_tokens(result)} tokens")
    return result

def manage_history_tokens(
    history: List[Dict[str, Any]],
    current_question: str,
    max_history_tokens: int = MAX_HISTORY_TOKENS
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Manage conversation history to stay within token limits.
    
    Args:
        history: List of conversation messages
        current_question: The current user question
        max_history_tokens: Maximum allowed tokens for history
    
    Returns:
        Tuple of (managed history, remaining tokens)
    """
    if not history:
        return [], max_history_tokens
    
    # Count tokens in the current question
    question_tokens = count_tokens(current_question)
    question_tokens = min(question_tokens, MAX_QUESTION_TOKENS)
    
    # Calculate available tokens for history
    available_tokens = max_history_tokens - question_tokens
    
    # If we have enough tokens for all history, return as is
    total_history_tokens = sum(count_tokens(msg["content"]) for msg in history)
    if total_history_tokens <= available_tokens:
        return history, available_tokens - total_history_tokens
    
    # We need to reduce history
    logger.info(f"Reducing history from {total_history_tokens} tokens to fit within {available_tokens} tokens")
    
    # Strategy: Keep most recent messages, drop oldest ones
    managed_history = []
    used_tokens = 0
    
    # Process messages in reverse order (newest first)
    for msg in reversed(history):
        msg_tokens = count_tokens(msg["content"])
        
        if used_tokens + msg_tokens <= available_tokens:
            managed_history.insert(0, msg)  # Insert at beginning to maintain order
            used_tokens += msg_tokens
        else:
            # We can't fit this message, so stop
            break
    
    # If we couldn't keep any messages, add a summary message
    if not managed_history and history:
        summary_msg = {
            "role": "system",
            "content": "Previous conversation history was too long and has been summarized. This is a new conversation about the same video."
        }
        summary_tokens = count_tokens(summary_msg["content"])
        managed_history = [summary_msg]
        used_tokens = summary_tokens
    
    remaining_tokens = available_tokens - used_tokens
    logger.info(f"Managed history to {used_tokens} tokens, {remaining_tokens} tokens remaining")
    return managed_history, remaining_tokens

def prepare_for_model(
    transcript: str,
    question: str,
    history: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Prepare transcript and history for the model, ensuring token limits are respected.
    
    Args:
        transcript: The video transcript
        question: The current user question
        history: Previous conversation history
    
    Returns:
        Tuple of (managed transcript, managed history)
    """
    # First, manage the history
    managed_history, _ = manage_history_tokens(history, question)
    
    # Then, truncate the transcript to fit within limits
    # We use a fixed transcript limit to ensure consistency
    transcript_limit = MAX_TRANSCRIPT_TOKENS
    managed_transcript = truncate_transcript(transcript, transcript_limit)
    
    # Log the total tokens being sent to the model
    total_input_tokens = (
        count_tokens(managed_transcript) +
        sum(count_tokens(msg["content"]) for msg in managed_history) +
        count_tokens(question)
    )
    logger.info(f"Total input tokens: {total_input_tokens}/{MAX_TOTAL_TOKENS} " +
                f"({(total_input_tokens/MAX_TOTAL_TOKENS)*100:.2f}% of limit)")
    
    return managed_transcript, managed_history

class GeminiClient:
    """
    Client for interacting with the Gemini API.
    
    This class provides methods for generating summaries and answering questions
    about YouTube videos using the Gemini API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key (optional, will use from settings if not provided)
        """
        settings = get_settings()
        self.api_key = api_key or settings.api_settings.GEMINI_API_KEY
        self.model_name = "gemini-2.5-flash-preview-04-17"  # Default model
        self.client = None
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Gemini client initialized with model: {self.model_name}")
        else:
            logger.warning("No Gemini API key provided. Client will not be functional.")
    
    @async_circuit_breaker(
        name="gemini_api",
        failure_threshold=5,
        recovery_time=60,
        half_open_timeout=30
    )
    async def generate_summary(
        self,
        transcript: str,
        summary_type: str,
        summary_length: str
    ) -> str:
        """
        Generate summary using Gemini API.
        
        Args:
            transcript: The video transcript text
            summary_type: The type of summary to generate
            summary_length: The desired length of the summary
        
        Returns:
            The generated summary text
        """
        if not self.client or not self.api_key:
            return "API key not configured. Unable to generate summary."
        
        if not transcript:
            return "No transcript available. Cannot generate summary."
        
        # Acquire semaphore to limit concurrent requests
        async with gemini_semaphore:
            try:
                # Truncate transcript if needed
                truncated_transcript = truncate_transcript(transcript, MAX_TRANSCRIPT_TOKENS)
                
                # Adjust prompt based on summary type and length
                length_words = {
                    "Short": "100-150 words",
                    "Medium": "200-300 words",
                    "Long": "400-600 words"
                }
                
                type_instruction = {
                    "Brief": "Create a concise overview",
                    "Detailed": "Create a comprehensive summary with key details",
                    "Key Point": "Extract and list the main points in bullet form",
                    "Chapters": "Divide the content into logical chapters with timestamps (if available) and provide a brief summary for each chapter"
                }
                
                prompt = f"""
                Based on the following transcript from a YouTube video, {type_instruction.get(summary_type, "create a summary")}.
                The summary should be approximately {length_words.get(summary_length, "200-300 words")} in length.
                Format the output in Markdown with appropriate headings, bullet points, and emphasis where needed.
                Do not include ```markdown at the start and end of the summary.
                IMPORTANT: Always generate the summary in English, regardless of the language of the transcript.
                
                {"For chapter-based summaries, identify logical sections in the content and create a chapter for each major topic or segment. Format each chapter with a clear heading that includes a timestamp (if you can identify it from the transcript) and a brief title. Under each chapter heading, provide a concise summary of that section." if summary_type == "Chapters" else ""}
                
                IMPORTANT: Exclude the following types of content from your summary:
                - Sponsor segments (paid promotions or advertisements)
                - Interaction reminders (like, subscribe, comment requests)
                - Unpaid/Self Promotion (merchandise, Patreon, personal projects)
                - Intro/outro animations or intermissions
                - End cards and credits
                - Preview/recap hooks for other content
                - Tangents, jokes, or skits unrelated to the main content
                - Non-essential music sections in non-music videos
                
                Focus only on the substantive, informative content of the video.
                
                TRANSCRIPT:
                {truncated_transcript}
                """
                
                # Create content using the API format
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)]
                    )
                ]
                
                # Configure generation parameters
                generate_content_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0,
                    ),
                    response_mime_type="text/plain",
                    max_output_tokens=MAX_OUTPUT_TOKENS
                )
                
                # Run in executor to avoid blocking
                response = await asyncio.to_thread(
                    self._generate_content,
                    contents,
                    generate_content_config
                )
                
                return response.text
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                return f"Failed to generate summary: {str(e)}"
    
    @async_circuit_breaker(
        name="gemini_qa",
        failure_threshold=5,
        recovery_time=60,
        half_open_timeout=30
    )
    async def generate_qa_response(
        self,
        transcript: str,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate answer to a question about a video using Gemini API.
        
        Args:
            transcript: The video transcript text
            question: The user's question
            history: Optional list of previous chat messages
        
        Returns:
            The generated answer text
        """
        if not self.client or not self.api_key:
            return "API key not configured. Unable to generate answer."
        
        if not transcript:
            return "No transcript available. Cannot answer questions about this video."
        
        # Acquire semaphore to limit concurrent requests
        async with gemini_semaphore:
            try:
                # Prepare history if provided
                history_for_model = []
                if history:
                    for msg in history:
                        history_for_model.append({
                            "role": "user" if msg["role"] == ChatMessageRole.USER else "model",
                            "content": msg["content"]
                        })
                
                # Apply token management
                managed_transcript, managed_history = prepare_for_model(
                    transcript, question, history_for_model
                )
                
                # Log token management results
                logger.info(f"Original transcript length: {count_tokens(transcript)} tokens")
                logger.info(f"Managed transcript length: {count_tokens(managed_transcript)} tokens")
                logger.info(f"Original history length: {len(history) if history else 0} messages")
                logger.info(f"Managed history length: {len(managed_history)} messages")
                
                # Prepare conversation history for the model
                contents = []
                
                # Add system message to instruct the model
                system_prompt = f"""
                You are an AI assistant that answers questions about YouTube videos based ONLY on the provided transcript.
                
                IMPORTANT RULES:
                1. ONLY answer based on information explicitly mentioned in the transcript.
                2. If the answer cannot be found in the transcript, clearly state that the information is not available in the video.
                3. Do not make up or infer information that is not directly stated in the transcript.
                4. Keep answers concise and to the point.
                5. If asked about timestamps or specific moments in the video, try to identify them from context clues in the transcript if possible.
                6. Format your responses in a clear, readable way using Markdown when appropriate.
                
                TRANSCRIPT:
                {managed_transcript}
                """
                
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=system_prompt)]
                ))
                
                # Add managed conversation history
                for msg in managed_history:
                    contents.append(types.Content(
                        role=msg["role"],
                        parts=[types.Part.from_text(text=msg["content"])]
                    ))
                
                # Add the current question
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=question)]
                ))
                
                # Configure generation parameters
                generate_content_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0,
                    ),
                    response_mime_type="text/plain",
                    max_output_tokens=MAX_OUTPUT_TOKENS
                )
                
                # Run in executor to avoid blocking
                response = await asyncio.to_thread(
                    self._generate_content,
                    contents,
                    generate_content_config
                )
                
                return response.text
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                return f"Failed to generate answer: {str(e)}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def _generate_content(
        self,
        contents: List[types.Content],
        config: types.GenerateContentConfig
    ) -> Any:
        """
        Generate content using the Gemini API with retry logic.
        
        Args:
            contents: List of content objects
            config: Generation configuration
        
        Returns:
            Gemini API response
        """
        if not self.client:
            raise ValueError("Gemini client not initialized")
        
        # Generate content
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )
        
        return response
    
    async def generate_streaming_response(
        self,
        transcript: str,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response to a question using Gemini API.
        
        Args:
            transcript: The video transcript text
            question: The user's question
            history: Optional list of previous chat messages
        
        Yields:
            Chunks of the generated response
        """
        if not self.client or not self.api_key:
            yield "API key not configured. Unable to generate answer."
            return
        
        if not transcript:
            yield "No transcript available. Cannot answer questions about this video."
            return
        
        # Acquire semaphore to limit concurrent requests
        async with gemini_semaphore:
            try:
                # Prepare history if provided
                history_for_model = []
                if history:
                    for msg in history:
                        history_for_model.append({
                            "role": "user" if msg["role"] == ChatMessageRole.USER else "model",
                            "content": msg["content"]
                        })
                
                # Apply token management
                managed_transcript, managed_history = prepare_for_model(
                    transcript, question, history_for_model
                )
                
                # Prepare conversation history for the model
                contents = []
                
                # Add system message to instruct the model
                system_prompt = f"""
                You are an AI assistant that answers questions about YouTube videos based ONLY on the provided transcript.
                
                IMPORTANT RULES:
                1. ONLY answer based on information explicitly mentioned in the transcript.
                2. If the answer cannot be found in the transcript, clearly state that the information is not available in the video.
                3. Do not make up or infer information that is not directly stated in the transcript.
                4. Keep answers concise and to the point.
                5. If asked about timestamps or specific moments in the video, try to identify them from context clues in the transcript if possible.
                6. Format your responses in a clear, readable way using Markdown when appropriate.
                
                TRANSCRIPT:
                {managed_transcript}
                """
                
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=system_prompt)]
                ))
                
                # Add managed conversation history
                for msg in managed_history:
                    contents.append(types.Content(
                        role=msg["role"],
                        parts=[types.Part.from_text(text=msg["content"])]
                    ))
                
                # Add the current question
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=question)]
                ))
                
                # Configure generation parameters
                generate_content_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0,
                    ),
                    response_mime_type="text/plain",
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    stream=True  # Enable streaming
                )
                
                # Generate streaming response
                stream = await asyncio.to_thread(
                    self._generate_streaming_content,
                    contents,
                    generate_content_config
                )
                
                # Process the stream
                for chunk in stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield chunk.text
            except Exception as e:
                logger.error(f"Error generating streaming response: {e}")
                yield f"Failed to generate answer: {str(e)}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def _generate_streaming_content(
        self,
        contents: List[types.Content],
        config: types.GenerateContentConfig
    ) -> Any:
        """
        Generate streaming content using the Gemini API with retry logic.
        
        Args:
            contents: List of content objects
            config: Generation configuration
        
        Returns:
            Gemini API streaming response
        """
        if not self.client:
            raise ValueError("Gemini client not initialized")
        
        # Generate streaming content
        return self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )

async def init_gemini_service() -> None:
    """Initialize the Gemini service."""
    global gemini_semaphore
    
    # Load settings
    settings = get_settings()
    max_connections = settings.resource_limits.GEMINI_API_MAX_CONNECTIONS
    
    # Update semaphore with configured value
    gemini_semaphore = asyncio.Semaphore(max_connections)
    
    logger.info(f"Gemini service initialized with max {max_connections} concurrent connections")
