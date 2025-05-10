"""
Core configuration module for the YouTube Summarizer API.

This module provides configuration settings for the application,
including environment-specific settings, feature flags, and resource limits.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field
import logging
from enum import Enum
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class EnvironmentType(str, Enum):
    """Environment types for the application."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class FeatureFlags(BaseSettings):
    """Feature flags for enabling/disabling functionality."""
    ENABLE_FULL_TRANSCRIPT: bool = Field(False, env="ENABLE_FULL_TRANSCRIPT")
    ENABLE_ADVANCED_SUMMARIZATION: bool = Field(False, env="ENABLE_ADVANCED_SUMMARIZATION")
    ENABLE_VIDEO_PREVIEW: bool = Field(False, env="ENABLE_VIDEO_PREVIEW")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class ResourceLimits(BaseSettings):
    """Resource limits for the application."""
    # Memory limits (in MB)
    MAX_MEMORY_USAGE: int = Field(480, env="MAX_MEMORY_USAGE")  # 93% of 512MB
    MEMORY_WARNING_THRESHOLD: int = Field(410, env="MEMORY_WARNING_THRESHOLD")  # 80% of 512MB
    
    # CPU limits
    MAX_CPU_USAGE: float = Field(0.09, env="MAX_CPU_USAGE")  # 90% of 0.1 cores
    
    # Redis memory limits (in MB)
    REDIS_MAX_MEMORY: int = Field(128, env="REDIS_MAX_MEMORY")
    
    # In-memory cache limits (in MB)
    IN_MEMORY_CACHE_SIZE: int = Field(10, env="IN_MEMORY_CACHE_SIZE")
    
    # Request limits
    MAX_URL_LENGTH: int = Field(2048, env="MAX_URL_LENGTH")
    MAX_REQUEST_PAYLOAD_KB: int = Field(10, env="MAX_REQUEST_PAYLOAD_KB")
    REQUEST_TIMEOUT_SECONDS: int = Field(60, env="REQUEST_TIMEOUT_SECONDS")
    
    # Connection limits
    YOUTUBE_API_MAX_CONNECTIONS: int = Field(5, env="YOUTUBE_API_MAX_CONNECTIONS")
    GEMINI_API_MAX_CONNECTIONS: int = Field(3, env="GEMINI_API_MAX_CONNECTIONS")
    
    # Rate limits
    GLOBAL_RATE_LIMIT_PER_MINUTE: int = Field(60, env="GLOBAL_RATE_LIMIT_PER_MINUTE")
    IP_RATE_LIMIT_PER_MINUTE: int = Field(10, env="IP_RATE_LIMIT_PER_MINUTE")
    
    # Throttling limits
    VIDEO_PROCESSING_MAX_CONCURRENT: int = Field(2, env="VIDEO_PROCESSING_MAX_CONCURRENT")
    AI_SUMMARIZATION_MAX_CONCURRENT: int = Field(1, env="AI_SUMMARIZATION_MAX_CONCURRENT")
    
    # Batch processing
    MAX_BATCH_SIZE: int = Field(5, env="MAX_BATCH_SIZE")
    BATCH_WINDOW_SECONDS: int = Field(2, env="BATCH_WINDOW_SECONDS")
    
    # Circuit breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    CIRCUIT_BREAKER_RECOVERY_SECONDS: int = Field(60, env="CIRCUIT_BREAKER_RECOVERY_SECONDS")
    CIRCUIT_BREAKER_HALF_OPEN_SECONDS: int = Field(30, env="CIRCUIT_BREAKER_HALF_OPEN_SECONDS")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class TimeoutSettings(BaseSettings):
    """Timeout settings for various operations."""
    HTTP_CLIENT_TIMEOUT: int = Field(30, env="HTTP_CLIENT_TIMEOUT")  # seconds
    DATABASE_OPERATION_TIMEOUT: int = Field(10, env="DATABASE_OPERATION_TIMEOUT")  # seconds
    REDIS_OPERATION_TIMEOUT: int = Field(5, env="REDIS_OPERATION_TIMEOUT")  # seconds
    YOUTUBE_API_TIMEOUT: int = Field(30, env="YOUTUBE_API_TIMEOUT")  # seconds
    GEMINI_API_TIMEOUT: int = Field(60, env="GEMINI_API_TIMEOUT")  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class CacheSettings(BaseSettings):
    """Cache settings for the application."""
    # TTL values in seconds
    VIDEO_METADATA_TTL: int = Field(86400, env="VIDEO_METADATA_TTL")  # 24 hours
    TRANSCRIPT_TTL: int = Field(172800, env="TRANSCRIPT_TTL")  # 48 hours
    SUMMARY_TTL: int = Field(259200, env="SUMMARY_TTL")  # 72 hours
    
    # Redis configuration
    REDIS_URL: str = Field("redis://localhost:6379", env="REDIS_URL")
    REDIS_MAX_CONNECTIONS: int = Field(10, env="REDIS_MAX_CONNECTIONS")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class DatabaseSettings(BaseSettings):
    """Database settings for the application."""
    MONGODB_URI: str = Field("mongodb://localhost:27017", env="MONGODB_URI")
    DATABASE_NAME: str = Field("youtube_summarizer", env="DATABASE_NAME")
    MAX_CONNECTIONS: int = Field(5, env="MONGODB_MAX_CONNECTIONS")
    CONNECTION_TIMEOUT: int = Field(5, env="MONGODB_CONNECTION_TIMEOUT")  # seconds
    IDLE_TIMEOUT: int = Field(60, env="MONGODB_IDLE_TIMEOUT")  # seconds
    QUERY_TIMEOUT: int = Field(10, env="MONGODB_QUERY_TIMEOUT")  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class APISettings(BaseSettings):
    """API settings for the application."""
    GEMINI_API_KEY: Optional[str] = Field(None, env="GEMINI_API_KEY")
    YOUTUBE_API_KEY: Optional[str] = Field(None, env="YOUTUBE_API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class Settings(BaseSettings):
    """Main settings class that combines all settings."""
    APP_NAME: str = Field("YouTube Summarizer API", env="APP_NAME")
    ENVIRONMENT: EnvironmentType = Field(EnvironmentType.DEVELOPMENT, env="ENVIRONMENT")
    DEBUG: bool = Field(False, env="DEBUG")
    
    # Subconfigurations
    feature_flags: FeatureFlags = FeatureFlags()
    resource_limits: ResourceLimits = ResourceLimits()
    timeout_settings: TimeoutSettings = TimeoutSettings()
    cache_settings: CacheSettings = CacheSettings()
    database_settings: DatabaseSettings = DatabaseSettings()
    api_settings: APISettings = APISettings()
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(["*"], env="CORS_ORIGINS")
    
    # Server settings
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")
    
    # Monitoring settings
    ENABLE_MEMORY_MONITORING: bool = Field(True, env="ENABLE_MEMORY_MONITORING")
    MEMORY_MONITORING_INTERVAL: int = Field(60, env="MEMORY_MONITORING_INTERVAL")  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()

def load_environment_settings(env_type: EnvironmentType = EnvironmentType.DEVELOPMENT) -> None:
    """
    Load environment-specific settings.
    
    Args:
        env_type: The environment type to load settings for
    """
    env_file = f"config/{env_type.value}.env"
    logger.info(f"Loading environment settings from {env_file}")
    
    # This will set environment variables from the file
    # which will be picked up by the Settings class
    os.environ["ENV_FILE"] = env_file

# Initialize settings
settings = get_settings()
