"""
MongoDB connection and utility functions for the YouTube Summarizer API.

This module provides optimized MongoDB connection handling and query utilities
designed for resource-constrained environments.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from bson import ObjectId

from ..core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Global MongoDB client
client: Optional[AsyncIOMotorClient] = None
db: Optional[AsyncIOMotorDatabase] = None

async def connect_to_mongodb() -> AsyncIOMotorClient:
    """
    Connect to MongoDB with optimized settings for resource-constrained environments.
    
    Returns:
        AsyncIOMotorClient: MongoDB client
    """
    global client, db
    
    settings = get_settings()
    db_settings = settings.database_settings
    
    # If already connected, return existing client
    if client is not None:
        return client
    
    try:
        # Configure MongoDB client with optimized settings
        client = AsyncIOMotorClient(
            db_settings.MONGODB_URI,
            maxPoolSize=db_settings.MAX_CONNECTIONS,  # Limit to 5 connections
            minPoolSize=1,
            maxIdleTimeMS=db_settings.IDLE_TIMEOUT * 1000,  # Convert to milliseconds
            connectTimeoutMS=db_settings.CONNECTION_TIMEOUT * 1000,  # Convert to milliseconds
            serverSelectionTimeoutMS=5000,  # 5 seconds for server selection
            socketTimeoutMS=10000,  # 10 seconds for socket operations
            waitQueueTimeoutMS=10000,  # 10 seconds for connection from pool
            retryWrites=True,
            w="majority",  # Write concern
        )
        
        # Test connection
        await client.admin.command('ping')
        logger.info(f"Connected to MongoDB at {db_settings.MONGODB_URI}")
        
        # Set database
        db = client[db_settings.DATABASE_NAME]
        
        # Create indexes
        await create_indexes()
        
        return client
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

async def close_mongodb_connection() -> None:
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        client = None
        logger.info("MongoDB connection closed")

async def get_database() -> AsyncIOMotorDatabase:
    """
    Get MongoDB database instance.
    
    Returns:
        AsyncIOMotorDatabase: MongoDB database
    """
    global db
    if db is None:
        await connect_to_mongodb()
    return db

async def create_indexes() -> None:
    """Create indexes for collections."""
    global db
    if db is None:
        return
    
    try:
        # Create indexes for video_chats collection
        await db.video_chats.create_index([("videoId", ASCENDING)], background=True)
        await db.video_chats.create_index([("userId", ASCENDING)], background=True)
        await db.video_chats.create_index([("createdAt", DESCENDING)], background=True)
        
        # Create indexes for summaries collection
        await db.summaries.create_index([("video_url", ASCENDING)], background=True)
        await db.summaries.create_index([("video_id", ASCENDING)], background=True)
        await db.summaries.create_index([("created_at", DESCENDING)], background=True)
        await db.summaries.create_index([("is_starred", ASCENDING)], background=True)
        await db.summaries.create_index(
            [
                ("summary_type", ASCENDING),
                ("summary_length", ASCENDING),
                ("video_id", ASCENDING)
            ],
            background=True
        )
        
        logger.info("MongoDB indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating MongoDB indexes: {e}")

async def execute_find_one(
    collection: str,
    query: Dict[str, Any],
    projection: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Execute a find_one query with timeout and projection optimization.
    
    Args:
        collection: Collection name
        query: Query filter
        projection: Field projection (only return specified fields)
        timeout: Query timeout in seconds
    
    Returns:
        Document or None if not found
    """
    global db
    if db is None:
        await connect_to_mongodb()
    
    settings = get_settings()
    timeout = timeout or settings.database_settings.QUERY_TIMEOUT
    
    try:
        # Use maxTimeMS to set query timeout
        result = await db[collection].find_one(
            filter=query,
            projection=projection,
            max_time_ms=timeout * 1000  # Convert to milliseconds
        )
        
        if result and '_id' in result:
            result['id'] = str(result.pop('_id'))
        
        return result
    except Exception as e:
        logger.error(f"Error executing find_one query on {collection}: {e}")
        return None

async def execute_find(
    collection: str,
    query: Dict[str, Any],
    projection: Optional[Dict[str, Any]] = None,
    sort: Optional[List[tuple]] = None,
    skip: int = 0,
    limit: int = 100,
    timeout: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Execute a find query with timeout and projection optimization.
    
    Args:
        collection: Collection name
        query: Query filter
        projection: Field projection (only return specified fields)
        sort: Sort specification
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        timeout: Query timeout in seconds
    
    Returns:
        List of documents
    """
    global db
    if db is None:
        await connect_to_mongodb()
    
    settings = get_settings()
    timeout = timeout or settings.database_settings.QUERY_TIMEOUT
    
    try:
        # Build the cursor
        cursor = db[collection].find(
            filter=query,
            projection=projection,
        )
        
        # Apply sort if specified
        if sort:
            cursor = cursor.sort(sort)
        
        # Apply pagination
        cursor = cursor.skip(skip).limit(limit)
        
        # Set timeout
        cursor = cursor.max_time_ms(timeout * 1000)  # Convert to milliseconds
        
        # Execute query and convert to list
        results = []
        async for doc in cursor:
            if '_id' in doc:
                doc['id'] = str(doc.pop('_id'))
            results.append(doc)
        
        return results
    except Exception as e:
        logger.error(f"Error executing find query on {collection}: {e}")
        return []

async def execute_count(
    collection: str,
    query: Dict[str, Any],
    timeout: Optional[int] = None
) -> int:
    """
    Execute a count query with timeout.
    
    Args:
        collection: Collection name
        query: Query filter
        timeout: Query timeout in seconds
    
    Returns:
        Document count
    """
    global db
    if db is None:
        await connect_to_mongodb()
    
    settings = get_settings()
    timeout = timeout or settings.database_settings.QUERY_TIMEOUT
    
    try:
        # Use maxTimeMS to set query timeout
        count = await db[collection].count_documents(
            filter=query,
            maxTimeMS=timeout * 1000  # Convert to milliseconds
        )
        return count
    except Exception as e:
        logger.error(f"Error executing count query on {collection}: {e}")
        return 0

async def execute_insert_one(
    collection: str,
    document: Dict[str, Any],
    timeout: Optional[int] = None
) -> Optional[str]:
    """
    Execute an insert_one operation with timeout.
    
    Args:
        collection: Collection name
        document: Document to insert
        timeout: Operation timeout in seconds
    
    Returns:
        Inserted document ID or None if failed
    """
    global db
    if db is None:
        await connect_to_mongodb()
    
    settings = get_settings()
    timeout = timeout or settings.database_settings.QUERY_TIMEOUT
    
    try:
        # Use maxTimeMS to set operation timeout
        result = await db[collection].insert_one(
            document=document,
            bypass_document_validation=False
        )
        
        if result.inserted_id:
            return str(result.inserted_id)
        return None
    except Exception as e:
        logger.error(f"Error executing insert_one operation on {collection}: {e}")
        return None

async def execute_update_one(
    collection: str,
    query: Dict[str, Any],
    update: Dict[str, Any],
    upsert: bool = False,
    timeout: Optional[int] = None
) -> bool:
    """
    Execute an update_one operation with timeout.
    
    Args:
        collection: Collection name
        query: Query filter
        update: Update operations
        upsert: Whether to insert if document doesn't exist
        timeout: Operation timeout in seconds
    
    Returns:
        True if successful, False otherwise
    """
    global db
    if db is None:
        await connect_to_mongodb()
    
    settings = get_settings()
    timeout = timeout or settings.database_settings.QUERY_TIMEOUT
    
    try:
        # Use maxTimeMS to set operation timeout
        result = await db[collection].update_one(
            filter=query,
            update=update,
            upsert=upsert,
            bypass_document_validation=False
        )
        
        return result.modified_count > 0 or (upsert and result.upserted_id is not None)
    except Exception as e:
        logger.error(f"Error executing update_one operation on {collection}: {e}")
        return False

async def execute_delete_one(
    collection: str,
    query: Dict[str, Any],
    timeout: Optional[int] = None
) -> bool:
    """
    Execute a delete_one operation with timeout.
    
    Args:
        collection: Collection name
        query: Query filter
        timeout: Operation timeout in seconds
    
    Returns:
        True if successful, False otherwise
    """
    global db
    if db is None:
        await connect_to_mongodb()
    
    settings = get_settings()
    timeout = timeout or settings.database_settings.QUERY_TIMEOUT
    
    try:
        # Use maxTimeMS to set operation timeout
        result = await db[collection].delete_one(
            filter=query
        )
        
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"Error executing delete_one operation on {collection}: {e}")
        return False

def parse_object_id(id_str: str) -> Optional[ObjectId]:
    """
    Parse a string into an ObjectId.
    
    Args:
        id_str: String representation of ObjectId
    
    Returns:
        ObjectId or None if invalid
    """
    try:
        return ObjectId(id_str)
    except Exception:
        return None
