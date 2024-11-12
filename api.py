import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import traceback
import json
from datetime import datetime, timezone

# Import from main.py
from main import (
    ConversationManager, 
    UserManager,
    memory_store,
    history,
    MessageHistory,
    CLI,
    display_status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Models
class StartConversationRequest(BaseModel):
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_name: Optional[str] = None

class StartConversationResponse(BaseModel):
    user_id: str
    thread_id: str
    initial_memories: str
    user_info: Optional[Dict[str, Any]] = None

class MessageRequest(BaseModel):
    message: str
    user_id: str
    thread_id: Optional[str] = None

class MessageResponse(BaseModel):
    response: str
    memories_updated: bool = False
    facts_updated: bool = False
    debug_info: Optional[Dict[str, Any]] = None

class MemoryRequest(BaseModel):
    user_id: str
    content: str = Field(..., description="The memory content to save")
    importance: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Importance level (1-5)"
    )
    metadata: Optional[Dict[str, Any]] = None

class FactRequest(BaseModel):
    fact: str = Field(..., description="The fact to save")
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reliability score (0-1)"
    )
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    min_importance: Optional[int] = None
    limit: Optional[int] = None

class UserListResponse(BaseModel):
    users: List[Dict[str, Any]]
    total_count: int
    active_count: int

class ThreadListResponse(BaseModel):
    threads: List[Dict[str, Any]]
    total_count: int
    active_count: int

# Initialize FastAPI app
app = FastAPI(
    title="AI Assistant API",
    description="API for interacting with the AI Assistant with memory and fact management",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
user_manager = UserManager()
conversation_managers: Dict[str, ConversationManager] = {}

# WebSocket connections store
active_connections: Dict[str, WebSocket] = {}

# Remove this import since we'll define it here
# from main import get_user_memories

# Add this function after the active_connections definition:

async def get_user_memories(user_id: str, min_importance: int = 0, limit: Optional[int] = None) -> str:
    """Get formatted user memories string"""
    try:
        memories = memory_store.get_memories(
            user_id=user_id,
            min_importance=min_importance,
            limit=limit
        )

        if not memories:
            return "No previous memories found."

        # Format memories into a readable string
        memory_text = "📚 User Memories:\n"
        for mem in memories:
            timestamp = datetime.fromisoformat(mem['timestamp']).strftime('%Y-%m-%d')
            memory_text += f"[{mem['importance']}★] {mem['content']} ({timestamp})\n"

        return memory_text

    except Exception as e:
        logger.error(f"Error getting user memories: {e}")
        return "Error retrieving memories."

# Then modify the start_conversation endpoint to use the async function:


# Also modify the memories endpoint:



async def get_or_create_conversation_manager(user_id: str) -> ConversationManager:
    """Get existing or create new conversation manager for user"""
    if user_id not in conversation_managers:
        conversation_managers[user_id] = ConversationManager()
    return conversation_managers[user_id]

@app.post("/users/create", response_model=Dict[str, str])
async def create_user(user_name: str):
    """Create a new user"""
    try:
        user_id = user_manager.create_user(user_name)
        return {"user_id": user_id, "message": f"User created successfully"}
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/list", response_model=UserListResponse)
async def list_users():
    """List all users"""
    try:
        users = user_manager.list_users()
        return UserListResponse(
            users=users,
            total_count=len(users),
            active_count=len([u for u in users if u.get('last_active')])
        )
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/threads", response_model=ThreadListResponse)
async def list_user_threads(user_id: str):
    """List all threads for a user"""
    try:
        threads = user_manager.get_user_threads(user_id)
        return ThreadListResponse(
            threads=threads,
            total_count=len(threads),
            active_count=len([t for t in threads if t.get('last_activity')])
        )
    except Exception as e:
        logger.error(f"Error listing threads: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation/start", response_model=StartConversationResponse)
async def start_conversation(request: StartConversationRequest):
    """Start or resume a conversation"""
    try:
        # Create new user if needed
        if not request.user_id and request.user_name:
            request.user_id = user_manager.create_user(request.user_name)

        manager = await get_or_create_conversation_manager(request.user_id)
        config = manager.start_conversation(
            user_id=request.user_id,
            thread_id=request.thread_id
        )

        user_info = user_manager.get_user_info(config["configurable"]["user_id"])
        user_memories = await get_user_memories(config["configurable"]["user_id"])

        return StartConversationResponse(
            user_id=config["configurable"]["user_id"],
            thread_id=config["configurable"]["thread_id"],
            initial_memories=user_memories,
            user_info=user_info
        )
    except Exception as e:
        logger.error(f"Error starting conversation: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/conversation/message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """Send a message in a conversation"""
    try:
        manager = await get_or_create_conversation_manager(request.user_id)

        # Process message and get response
        response = manager.process_message(request.message)

        # Get debug info
        debug_info = {
            "user_id": request.user_id,
            "thread_id": manager.current_thread_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_processed": True
        }

        return MessageResponse(
            response=response,
            memories_updated=True,  # You might want to make this dynamic
            facts_updated=True,     # You might want to make this dynamic
            debug_info=debug_info
        )
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time chat"""
    try:
        await websocket.accept()
        active_connections[user_id] = websocket

        manager = await get_or_create_conversation_manager(user_id)

        while True:
            try:
                data = await websocket.receive_json()
                message = data.get("message")

                if message:
                    response = manager.process_message(message)
                    await websocket.send_json({
                        "response": response,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
    finally:
        if user_id in active_connections:
            del active_connections[user_id]

@app.get("/memories", response_model=Dict[str, Any])
async def get_user_memories_endpoint(
    user_id: str,
    min_importance: Optional[int] = None,
    limit: Optional[int] = None
):
    """Get memories for a user"""
    try:
        memories = await get_user_memories(
            user_id=user_id,
            min_importance=min_importance if min_importance is not None else 0,
            limit=limit
        )
        return {
            "memories": memories,
            "formatted": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories/search", response_model=Dict[str, Any])
async def search_memories_endpoint(request: SearchRequest):
    """Search user memories"""
    try:
        results = memory_store.search_memories(
            user_id=request.user_id,
            query=request.query,
            min_importance=request.min_importance
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/save")
async def save_memory_endpoint(request: MemoryRequest):
    """Save a new memory"""
    try:
        memory_store.save_memory(
            user_id=request.user_id,
            memory=request.content,
            metadata=request.metadata,
            importance=request.importance
        )
        return {"status": "success", "message": "Memory saved successfully"}
    except Exception as e:
        logger.error(f"Error saving memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/facts/save")
async def save_fact_endpoint(request: FactRequest):
    """Save a new fact"""
    try:
        memory_store.save_fact(
            fact=request.fact,
            reliability_score=request.score,
            additional_info=request.metadata
        )
        return {"status": "success", "message": "Fact saved successfully"}
    except Exception as e:
        logger.error(f"Error saving fact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/facts/search")
async def search_facts_endpoint(query: str, min_score: Optional[float] = None):
    """Search facts"""
    try:
        results = memory_store.search_facts(query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching facts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os

    # Determine if we're in development mode
    is_dev = os.getenv("DEVELOPMENT", "false").lower() == "true"

    # Get port from environment or default to 80 for prod, 8000 for dev
    port = int(os.getenv("PORT", 8000 if is_dev else 80))

    if is_dev:
        # Development configuration
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level="debug"
        )
    else:
        # Production configuration
        config = uvicorn.Config(
            "api:app",
            host="0.0.0.0",
            port=port,
            log_level="info",
            reload=False,
            workers=1
        )
        server = uvicorn.Server(config)
        server.run()