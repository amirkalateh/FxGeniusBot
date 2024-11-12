from re import search
from typing import Annotated, Any, Dict, List, Union, Optional, Sequence, Literal
from typing_extensions import TypedDict
from datetime import datetime, timezone
import logging
import os
import sys
import json
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (HumanMessage, SystemMessage, AIMessage,
                                     BaseMessage, ToolMessage, FunctionMessage)
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser

# LangGraph imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import (create_react_agent, InjectedStore, ToolNode,
                                tools_condition)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph

# Database imports
from sqlalchemy import (create_engine, Column, String, Integer, Text, DateTime,
                        JSON, desc, func)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from pydantic import BaseModel, Field, ValidationError

from contextlib import contextmanager
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger(__name__)

# Database initialization with better error handling
try:
    # Get database URL
    database_url = os.getenv("DATABASE_URL") or "sqlite:///example.db"
    logger.info(
        f"Initializing database connection to: {database_url.split('@')[-1]}"
    )  # Log only the non-sensitive part

    # Create engine with proper settings
    engine = create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=5,
        pool_pre_ping=True,
        pool_recycle=3600,
        max_overflow=10,
        echo=False  # Set to True for SQL debugging
    )

    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        logger.info("Database connection test successful")

    # Create session factory
    SessionLocal = sessionmaker(bind=engine,
                                autocommit=False,
                                autoflush=False,
                                expire_on_commit=False)

    # Create declarative base using the new style
    from sqlalchemy.orm import declarative_base
    Base = declarative_base()

    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")

except Exception as e:
    logger.error(f"Database initialization failed: {str(e)}")
    sys.exit(1)


# Context manager for database sessions
@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()


# Function to verify database connectivity
def verify_database():
    try:
        # Test session creation and basic query
        with get_session() as session:
            # Try to execute a simple query
            session.execute(text("SELECT 1"))
            logger.info("Database verification successful")
            return True
    except Exception as e:
        logger.error(f"Database verification failed: {str(e)}")
        return False


# Function to initialize database
def initialize_database():
    """Initialize database and create tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")

        # Verify connectivity
        if not verify_database():
            raise Exception("Database verification failed")

        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return False


# Load environment variables
load_dotenv()

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger(__name__)


# Enhanced database connection handling
def get_database_url():
    """Get database URL with fallback to individual components"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        required_vars = [
            "PGDATABASE", "PGHOST", "PGPORT", "PGUSER", "PGPASSWORD"
        ]
        values = {var: os.getenv(var) for var in required_vars}

        if not all(values.values()):
            missing = [var for var, val in values.items() if not val]
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return f"postgresql+psycopg2://{values['PGUSER']}:{values['PGPASSWORD']}@{values['PGHOST']}:{values['PGPORT']}/{values['PGDATABASE']}"

    return database_url


# Initialize database with connection pooling and retry logic
@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=4, max=10))
def create_db_engine():
    """Create database engine with connection pooling and retry logic"""
    return create_engine(get_database_url(),
                         poolclass=QueuePool,
                         pool_size=5,
                         max_overflow=10,
                         pool_timeout=30,
                         pool_pre_ping=True,
                         echo=False)


# Initialize database components
try:
    engine = create_db_engine()
    SessionLocal = sessionmaker(bind=engine)
    Base = declarative_base()
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    sys.exit(1)


# Database Models with enhanced functionality
class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    thread_id = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True),
                       default=datetime.utcnow,
                       nullable=False)
    extra_metadata = Column(JSON, nullable=True)

    def to_message_object(
            self
    ) -> Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]:
        """Convert database message to LangChain message object"""
        kwargs = {
            "additional_kwargs": {
                "timestamp": self.timestamp.isoformat(),
                **(self.extra_metadata or {})
            }
        }

        message_types = {
            "system": SystemMessage,
            "human": HumanMessage,
            "ai": AIMessage,
            "tool": ToolMessage
        }

        MessageClass = message_types.get(self.type)
        if not MessageClass:
            raise ValueError(f"Unknown message type: {self.type}")

        return MessageClass(content=self.content, **kwargs)


class MemoryModelDB(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    content = Column(Text, nullable=False)
    importance = Column(Integer, default=1, nullable=False)
    timestamp = Column(DateTime(timezone=True),
                       default=datetime.utcnow,
                       nullable=False)
    extra_metadata = Column(JSON, nullable=True)

    @classmethod
    def get_recent_important(cls, session, user_id: str, limit: int = 5):
        """Get recent important memories"""
        return (session.query(cls).filter_by(user_id=user_id).order_by(
            cls.importance.desc(), cls.timestamp.desc()).limit(limit).all())


class Fact(Base):
    __tablename__ = "facts"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=False)
    user_id = Column(String, index=True, nullable=True)
    score = Column(Integer, default=0)
    timestamp = Column(DateTime(timezone=True),
                       default=datetime.utcnow,
                       nullable=False)
    extra_metadata = Column(JSON, nullable=True)

    @classmethod
    def get_verified_facts(cls, session, query: str, min_score: int = 0):
        """Get verified facts matching query"""
        return (session.query(cls).filter(
            cls.score >= min_score,
            func.lower(cls.question).contains(func.lower(query))).order_by(
                cls.score.desc()).all())


# Create tables
Base.metadata.create_all(bind=engine)


# Session management with context
class DBContext:

    def __init__(self):
        self.session = None

    def __enter__(self):
        self.session = SessionLocal()
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                self.session.rollback()
            self.session.close()


def get_db():
    """Get database session with context management"""
    return DBContext()


# Custom Types for Type Safety
class MessageDict(TypedDict):
    type: str
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]]


class MemoryEntry(TypedDict):
    content: str
    timestamp: str
    importance: int
    metadata: Optional[Dict[str, Any]]


class FactEntry(TypedDict):
    question: str
    answer: str
    score: int
    timestamp: str
    metadata: Optional[Dict[str, Any]]


# Enhanced Message History Handler
class MessageHistory:
    """Enhanced Message History with error handling and validation"""

    def __init__(self):
        self.messages: List[Union[SystemMessage, HumanMessage, AIMessage]] = []
        self.current_user_id: Optional[str] = None
        self.current_thread_id: Optional[str] = None

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10))
    def load_thread(
            self, user_id: str, thread_id: str
    ) -> List[Union[SystemMessage, HumanMessage, AIMessage]]:
        """Load conversation thread with retry logic"""
        self.current_user_id = user_id
        self.current_thread_id = thread_id
        self.messages = []

        try:
            with get_db() as session:
                db_messages = (session.query(Message).filter_by(
                    user_id=user_id,
                    thread_id=thread_id).order_by(Message.timestamp).all())

                if db_messages:
                    self.messages = [
                        msg.to_message_object() for msg in db_messages
                    ]
                else:
                    # Initialize with default system message
                    system_message = SystemMessage(
                        content=
                        """You are a highly capable AI assistant with access to user memories and real-time information.

KEY CAPABILITIES:
1. Access and utilize user context for personalized responses
2. Search for real-time information when needed
3. Save important user information and verified facts
4. Maintain conversation context and history

OPERATIONAL GUIDELINES:
1. Prioritize using existing knowledge before searching
2. Save user-specific information using save_memory tool
3. Save verified facts and discoveries using save_fact tool
4. Use search tool for real-time verification when uncertain
5. Maintain a confident, direct communication style
6. Focus on accuracy and efficiency in responses

Remember to stay within your knowledge cutoff date and verify uncertain information through search.""",
                        additional_kwargs={
                            "timestamp":
                            datetime.now(timezone.utc).isoformat()
                        })
                    self.add_message(system_message, save_to_db=True)

                logger.info(
                    f"Loaded thread {thread_id} for user {user_id} with {len(self.messages)} messages"
                )
                return self.messages

        except Exception as e:
            logger.error(f"Error loading thread: {e}")
            raise

    def add_message(self,
                    message: Union[SystemMessage, HumanMessage, AIMessage,
                                   ToolMessage],
                    save_to_db: bool = True,
                    process_memory: bool = True) -> None:
        """Add message with enhanced error handling and memory processing"""
        if not self.current_user_id or not self.current_thread_id:
            raise ValueError("No active thread - call load_thread first")

        try:
            # Handle system messages specially
            if isinstance(message, SystemMessage):
                self.messages = [
                    m for m in self.messages
                    if not isinstance(m, SystemMessage)
                ]
                self.messages.insert(0, message)
            else:
                self.messages.append(message)

            # Process AI messages for potential memory storage
            if process_memory and isinstance(message, AIMessage):
                self._process_ai_memory(message)

            if save_to_db:
                self._save_message_to_db(message)

        except Exception as e:
            logger.error(f"Error adding message: {e}")
            raise

    def _save_message_to_db(
        self, message: Union[SystemMessage, HumanMessage, AIMessage,
                             ToolMessage]
    ) -> None:
        """Save message to database with enhanced error handling"""
        try:
            with get_db() as session:
                msg_type = {
                    SystemMessage: "system",
                    HumanMessage: "human",
                    AIMessage: "ai",
                    ToolMessage: "tool"
                }.get(type(message))

                if not msg_type:
                    raise ValueError(
                        f"Unsupported message type: {type(message)}")

                # Handle content serialization
                content = (message.content if isinstance(message.content, str)
                           else json.dumps(message.content))

                # Extract timestamp from additional_kwargs or use current time
                timestamp_str = message.additional_kwargs.get("timestamp")
                try:
                    timestamp = (datetime.fromisoformat(timestamp_str) if
                                 timestamp_str else datetime.now(timezone.utc))
                except ValueError:
                    timestamp = datetime.now(timezone.utc)

                db_message = Message(user_id=self.current_user_id,
                                     thread_id=self.current_thread_id,
                                     type=msg_type,
                                     content=content,
                                     timestamp=timestamp,
                                     extra_metadata=message.additional_kwargs)

                session.add(db_message)
                session.commit()
                logger.info(f"Saved {msg_type} message to database")

        except Exception as e:
            logger.error(f"Database error saving message: {e}")
            raise

    def _process_ai_memory(self, message: AIMessage) -> None:
        """Process AI messages for memory triggers and fact extraction"""
        if not isinstance(message.content, str):
            return

        content_lower = message.content.lower()

        # Memory triggers that indicate actual user information
        memory_triggers = [
            "remember that",
            "important to note",
            "don't forget",
            "keep in mind",
            "worth remembering",
            "saved your",  # For catching "saved your preference" etc.
            "noted that you"
        ]

        # Fact triggers that indicate actual discovered information
        fact_triggers = [
            "according to sources", "research shows", "i found that",
            "discovered that", "search revealed", "latest version is",
            "recently released"
        ]

        try:
            # Don't save confirmation messages as facts
            if any(phrase in content_lower for phrase in [
                    "i've saved", "i have saved", "saved as a fact",
                    "saved it as", "great, i've", "please let me know"
            ]):
                return

            if any(trigger in content_lower for trigger in fact_triggers):
                # Extract and save fact, but only if it contains actual information
                memory_store.save_fact(fact=message.content,
                                       reliability_score=0.8,
                                       additional_info={
                                           "source":
                                           "ai_message",
                                           "thread_id":
                                           self.current_thread_id,
                                           "timestamp":
                                           datetime.now(
                                               timezone.utc).isoformat()
                                       })

            elif any(trigger in content_lower for trigger in memory_triggers):
                # Save as user memory
                memory_store.save_memory(self.current_user_id,
                                         message.content,
                                         metadata={
                                             "type": "auto_saved",
                                             "source": "ai_message",
                                             "thread_id":
                                             self.current_thread_id
                                         },
                                         importance=3)

        except Exception as e:
            logger.error(f"Error processing AI memory: {e}")
            # Don't raise - allow conversation to continue even if memory processing fails

    def get_recent_messages(
        self,
        limit: int = 10
    ) -> List[Union[SystemMessage, HumanMessage, AIMessage]]:
        """Get most recent messages with system context"""
        if not self.messages:
            return []

        # Always include system message if present
        system_msgs = [
            m for m in self.messages if isinstance(m, SystemMessage)
        ]
        other_msgs = [
            m for m in self.messages if not isinstance(m, SystemMessage)
        ]

        return system_msgs + other_msgs[-limit:]

    def get_thread_summary(self) -> str:
        """Generate thread summary with error handling"""
        try:
            if not self.messages:
                return "No messages in current thread."

            msg_count = len(self.messages)
            msg_types = {}

            for msg in self.messages:
                msg_type = type(msg).__name__
                msg_types[msg_type] = msg_types.get(msg_type, 0) + 1

            last_msg = self.messages[-1]
            last_time = last_msg.additional_kwargs.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat())

            summary = (
                f"Thread contains {msg_count} messages\n"
                f"Message types: {', '.join(f'{k}: {v}' for k, v in msg_types.items())}\n"
                f"Last activity: {last_time}")

            return summary

        except Exception as e:
            logger.error(f"Error generating thread summary: {e}")
            return "Error generating thread summary"


class UserMemoryStore:
    """Enhanced memory storage system with error handling and optimization"""

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10))
    def save_memory(
        self,
        user_id: str,
        memory: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: int = 1,
    ) -> None:
        """Save memory with retry logic and validation"""
        if not user_id or not memory:
            raise ValueError("User ID and memory content are required")

        try:
            with get_db() as session:
                # Check for duplicate memories to avoid redundancy
                existing = (session.query(MemoryModelDB).filter_by(
                    user_id=user_id, content=memory).first())

                if existing:
                    # Update importance if new importance is higher
                    if importance > existing.importance:
                        existing.importance = importance
                        existing.extra_metadata = {
                            **(existing.extra_metadata or {}),
                            **(metadata or {}), "updated_at":
                            datetime.now(timezone.utc).isoformat()
                        }
                        session.commit()
                        logger.info(
                            f"Updated existing memory importance: {memory}")
                    return

                # Create new memory
                db_memory = MemoryModelDB(user_id=user_id,
                                          content=memory,
                                          importance=importance,
                                          timestamp=datetime.now(timezone.utc),
                                          extra_metadata=metadata)
                session.add(db_memory)
                session.commit()
                logger.info(f"Saved new memory: {memory}")

        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            raise

    @retry(stop=stop_after_attempt(2))
    def get_memories(self,
                     user_id: str,
                     min_importance: int = 0,
                     limit: Optional[int] = None) -> List[MemoryEntry]:
        """Get memories with filtering and sorting"""
        try:
            with get_db() as session:
                query = (session.query(MemoryModelDB).filter(
                    MemoryModelDB.user_id == user_id, MemoryModelDB.importance
                    >= min_importance).order_by(
                        MemoryModelDB.importance.desc(),
                        MemoryModelDB.timestamp.desc()))

                if limit:
                    query = query.limit(limit)

                memories = query.all()

                return [{
                    "content": mem.content,
                    "timestamp": mem.timestamp.isoformat(),
                    "importance": mem.importance,
                    "metadata": mem.extra_metadata
                } for mem in memories]

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            raise

    def search_memories(self,
                        user_id: str,
                        query: str,
                        min_importance: int = 0) -> List[MemoryEntry]:
        """Search memories with importance filtering"""
        try:
            with get_db() as session:
                memories = (session.query(MemoryModelDB).filter(
                    MemoryModelDB.user_id == user_id,
                    MemoryModelDB.importance >= min_importance,
                    MemoryModelDB.content.ilike(f"%{query}%")).order_by(
                        MemoryModelDB.importance.desc(),
                        MemoryModelDB.timestamp.desc()).all())

                return [{
                    "content": mem.content,
                    "timestamp": mem.timestamp.isoformat(),
                    "importance": mem.importance,
                    "metadata": mem.extra_metadata
                } for mem in memories]

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def save_fact(self,
                  fact: str,
                  reliability_score: float = 0,
                  additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Save verified fact with deduplication"""
        try:
            with get_db() as session:
                # Check for similar existing facts
                similar_facts = (session.query(Fact).filter(
                    Fact.answer.ilike(f"%{fact[:50]}%")).all())

                # Update existing fact if similar
                for existing in similar_facts:
                    similarity_score = self._calculate_similarity(
                        existing.answer, fact)
                    if similarity_score > 0.8:  # High similarity threshold
                        if reliability_score > existing.score:
                            existing.answer = fact
                            existing.score = reliability_score
                            existing.extra_metadata = {
                                **(existing.extra_metadata or {}),
                                **(additional_info or {}), "updated_at":
                                datetime.now(timezone.utc).isoformat()
                            }
                            session.commit()
                            logger.info(f"Updated existing fact: {fact}")
                        return

                # Create new fact
                new_fact = Fact(answer=fact,
                                score=reliability_score,
                                timestamp=datetime.now(timezone.utc),
                                extra_metadata=additional_info)
                session.add(new_fact)
                session.commit()
                logger.info(f"Saved new fact: {fact}")

        except Exception as e:
            logger.error(f"Error saving fact: {e}")
            raise

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity score"""
        # Simple implementation - could be enhanced with more sophisticated algorithms
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0

    def search_facts(self, query: str, min_score: int = 0) -> List[str]:
        """Search verified facts"""
        try:
            with get_db() as session:
                facts = (session.query(Fact).filter(
                    Fact.score >= min_score,
                    func.lower(Fact.answer).contains(
                        func.lower(query))).order_by(Fact.score.desc()).all())
                return [
                    f"📖 {fact.answer} (Score: {fact.score})" for fact in facts
                ]
        except Exception as e:
            logger.error(f"Error searching facts: {e}")
            return []


# Initialize stores
history = MessageHistory()
memory_store = UserMemoryStore()
store = InMemoryStore()


# Tool Definitions with Enhanced Error Handling
class MemoryInput(BaseModel):
    """Schema for memory input validation"""
    content: str = Field(..., description="The memory content to save")
    importance: int = Field(default=1,
                            ge=1,
                            le=5,
                            description="Importance level (1-5)")


class FactInput(BaseModel):
    """Schema for fact input validation"""
    fact: str = Field(..., description="The fact to save")
    score: float = Field(default=0.0,
                         ge=0.0,
                         le=1.0,
                         description="Reliability score (0-1)")


@tool
def save_memory(input: MemoryInput, *, config: RunnableConfig,
                store: Annotated[Any, InjectedStore()]) -> str:
    """
    Save important information about the user.
    Args:
        input: MemoryInput containing content and importance level
    Returns:
        Confirmation message
    """
    try:
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "❌ Error: No user ID provided"

        metadata = {
            "importance": input.importance,
            "source": "explicit_save",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        memory_store.save_memory(user_id,
                                 input.content,
                                 metadata,
                                 importance=input.importance)

        return f"✅ Saved memory (importance: {input.importance}★): {input.content}"

    except ValidationError as e:
        return f"❌ Validation error: {str(e)}"
    except Exception as e:
        logger.error(f"Error in save_memory tool: {e}")
        return f"❌ Failed to save memory: {str(e)}"


@tool
def save_fact(input: FactInput, *, config: RunnableConfig,
              store: Annotated[Any, InjectedStore()]) -> str:
    """
    Save a verified fact with reliability score.
    Args:
        input: FactInput containing fact and reliability score
    Returns:
        Confirmation message
    """
    try:
        memory_store.save_fact(fact=input.fact,
                               reliability_score=input.score,
                               additional_info={
                                   "source":
                                   "tool_save",
                                   "timestamp":
                                   datetime.now(timezone.utc).isoformat()
                               })

        return f"✅ Saved fact (reliability: {input.score:.2f}): {input.fact}"

    except ValidationError as e:
        return f"❌ Validation error: {str(e)}"
    except Exception as e:
        logger.error(f"Error in save_fact tool: {e}")
        return f"❌ Failed to save fact: {str(e)}"


@tool
def search_user_info(query: str, *, config: RunnableConfig,
                     store: Annotated[Any, InjectedStore()]) -> str:
    """
    Search through user's stored memories.
    Args:
        query: Search term
    Returns:
        Formatted search results
    """
    try:
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "❌ Error: No user ID provided"

        results = memory_store.search_memories(user_id, query)
        if not results:
            return f"No memories found matching '{query}'"

        response = f"🔍 Found {len(results)} relevant memories:\n"
        for memory in results:
            importance = memory.get("importance", 1)
            timestamp = datetime.fromisoformat(
                memory["timestamp"]).strftime("%Y-%m-%d")
            response += f"- [{importance}★] {memory['content']} ({timestamp})\n"

        return response

    except Exception as e:
        logger.error(f"Error in search_user_info tool: {e}")
        return f"❌ Search failed: {str(e)}"


class AgentState(TypedDict):
    """Agent state definition with message history"""
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    remaining_steps: Optional[int]


def prepare_model_inputs(
    state: Dict,
    config: RunnableConfig,
    store: Optional[BaseStore] = None
) -> List[Union[SystemMessage, HumanMessage, AIMessage]]:
    """
    Prepare messages for the model including user context and memories
    """
    try:
        # Get user context
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            logger.warning("No user_id found in config")
            return state["messages"]

        # Get user memories
        memories = memory_store.get_memories(
            user_id=user_id,
            min_importance=
            2,  # Only include moderately to highly important memories
            limit=10  # Limit to most recent/important memories
        )

        # Format memories for context
        memory_text = "No previous information about this user."
        if memories:
            memory_text = "📚 Key user context:\n" + "\n".join(
                f"[{mem['importance']}★] {mem['content']}" for mem in memories)

        # Create enhanced system message
        system_message = SystemMessage(
            content=
            f"""You are a highly capable AI assistant with memory and real-time information access.

CURRENT USER CONTEXT:
{memory_text}
### IF USER TOLD YOU ABOUT HIS/HER INFORMATIONS, CALL BY HIS/HER NAME, AND SHOW HER/HIM YOU REMEMBER EVERY IMPORTANT THINK WELL. ALSO YOU CAN ASK USER IF YOU DONT HAVE THIS DATA TO BE COMFORTABLE AND SHARE FOR A BETTER AND SMARTER EXPERIENCE


CORE CAPABILITIES & GUIDELINES:

1. Memory Management:
   - Save important user information (preferences, facts, context)
   - Use save_memory for user-specific information
   - Use save_fact for verified general knowledge
   - Record information with appropriate importance (1-5)

2. Information Retrieval:
   - Use existing knowledge first
   - Search for real-time information when needed. TODAY IS 10 NOV 2024
   - Verify uncertain information before stating as fact
   - Use search_user_info to recall user details
   -VERY IMPORTANT: AFTER EACH SEARCH, SAVE IT WITH save_fact TOOL NICE AND CLEAR . IT WILL BE YOUR KNOWLEDGEBASE IN A MIDTERM.

3. Response Strategy:
   - Be direct and confident when information is certain
   - Clearly indicate when information needs verification
   - Maintain conversation context
   - Use appropriate formality based on user interaction history

4. Error Handling:
   - Gracefully handle missing or uncertain information
   - Request clarification when needed
   - Maintain conversation flow even if tool calls fail

Remember: Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. Prioritize accurate, relevant responses while maintaining context awareness.
""",
            additional_kwargs={
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        # Get conversation history
        messages = state["messages"]

        # Replace or insert system message
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = system_message
        else:
            messages.insert(0, system_message)

        return messages

    except Exception as e:
        logger.error(f"Error preparing model inputs: {e}")
        # Return original messages if preparation fails
        return state["messages"]


def create_enhanced_agent(temperature: float = 0.3,
                          debug: bool = False) -> CompiledGraph:
    """
    Create an enhanced agent with error handling and retry logic
    """
    try:
        # Initialize memory components
        memory = MemorySaver()

        # Initialize model with error handling
        model = ChatAnthropic(model_name="claude-3-haiku-20240307",
                              temperature=temperature,
                              anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                              max_retries=3,
                              timeout=40)

        # Initialize search tool with error handling
        search = TavilySearchResults(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=5,
        )

        # Define tools with proper error handling
        tools = [search, save_memory, save_fact, search_user_info]

        # Create agent with enhanced configuration
        return create_react_agent(model=model,
                                  tools=tools,
                                  checkpointer=memory,
                                  store=store,
                                  state_modifier=prepare_model_inputs,
                                  debug=debug)

    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise


class ConversationManager:
    """Enhanced conversation management with proper thread handling"""

    def __init__(self, temperature: float = 0.3, debug: bool = False):
        self.agent = create_enhanced_agent(temperature=temperature,
                                           debug=debug)
        self.config = {}
        self.error_count = 0
        self.max_errors = 3
        self.current_user_id = None
        self.current_thread_id = None
        self.user_manager = UserManager()

    def start_conversation(self,
                           user_id: str,
                           thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Initialize or resume conversation with thread management"""
        try:
            self.current_user_id = user_id

            # Force new thread creation if none provided
            if not thread_id:
                self.current_thread_id = f"thread_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            else:
                # Verify thread exists and belongs to user
                with get_db() as session:
                    thread_exists = (session.query(Message).filter_by(
                        user_id=user_id,
                        thread_id=thread_id).first()) is not None

                    if thread_exists:
                        self.current_thread_id = thread_id
                    else:
                        # Create new thread if specified thread doesn't exist
                        self.current_thread_id = f"thread_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                        logger.warning(
                            f"Thread {thread_id} not found for user {user_id}, creating new thread"
                        )

            # Setup configuration
            self.config = {
                "configurable": {
                    "thread_id": self.current_thread_id,
                    "user_id": self.current_user_id,
                }
            }

            # Clear any existing messages before loading new thread
            history.messages = []

            # Load conversation history for the specific thread
            history.load_thread(self.current_user_id, self.current_thread_id)

            # Log session information
            logger.info(
                f"Started conversation - User: {self.current_user_id}, Thread: {self.current_thread_id}"
            )

            return self.config

        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
            raise

    def reset_conversation(self):
        """Reset conversation state after errors"""
        try:
            self.error_count = 0
            history.load_thread(self.current_user_id, self.current_thread_id)
            logger.info(f"Reset conversation for user {self.current_user_id}")
        except Exception as e:
            logger.error(f"Error resetting conversation: {e}")

    def end_conversation(self):
        """Clean up conversation resources"""
        try:
            logger.info(f"Ending conversation for user {self.current_user_id}")
            self.current_user_id = None
            self.current_thread_id = None
            self.error_count = 0
        except Exception as e:
            logger.error(f"Error ending conversation: {e}")

    def process_message(self, user_input: str) -> str:
        """Process user message with enhanced error handling"""
        if not user_input.strip():
            return "Please provide a message."

        try:
            # Create user message
            user_message = HumanMessage(content=user_input,
                                        additional_kwargs={
                                            "timestamp":
                                            datetime.now(
                                                timezone.utc).isoformat()
                                        })

            # Add to history
            history.add_message(user_message)

            # Process message through agent
            response_parts = []
            tool_outputs = []

            for chunk in self.agent.stream({"messages": [user_message]},
                                           self.config):
                # Process agent response
                if "agent" in chunk:
                    agent_message = chunk["agent"].get("messages", [""])[0]
                    if isinstance(agent_message.content, str):
                        response_parts.append(agent_message.content)
                        history.add_message(agent_message, save_to_db=True)

                # Process tool outputs
                if "tools" in chunk:
                    tool_outputs.extend(chunk["tools"].get("messages", []))

            # Combine responses
            final_response = " ".join(response_parts) if response_parts else ""

            # Reset error count on successful processing
            self.error_count = 0

            return final_response

        except Exception as e:
            self.error_count += 1
            logger.error(
                f"Error processing message (attempt {self.error_count}): {e}")

            if self.error_count >= self.max_errors:
                logger.error("Max errors reached, resetting conversation")
                self.reset_conversation()
                return "I'm having trouble processing messages. The conversation has been reset. Please try again."

            # Provide appropriate error message based on error type
            if "invalid_request_error" in str(e):
                return "I encountered an error in processing. Could you rephrase your message?"
            elif "timeout" in str(e).lower():
                return "The response took too long. Could you try a simpler question?"
            else:
                return "I encountered an unexpected error. Please try again or rephrase your message."


class UserManager:
    """Manages user creation and selection with proper thread handling"""

    def __init__(self):
        self.db_context = get_db()

    def list_users(self) -> List[Dict[str, Any]]:
        """Get list of all users with their stats"""
        try:
            with get_db() as session:
                # Get unique users with their latest activity
                user_stats = (session.query(
                    Message.user_id,
                    func.max(Message.timestamp).label('last_active'),
                    func.count(Message.id).label('message_count')).group_by(
                        Message.user_id).order_by(
                            func.max(Message.timestamp).desc()).all())

                users = []
                for stat in user_stats:
                    # Get memory count for each user
                    memory_count = (session.query(MemoryModelDB).filter_by(
                        user_id=stat.user_id).count())

                    users.append({
                        "user_id": stat.user_id,
                        "last_active": stat.last_active,
                        "message_count": stat.message_count,
                        "memory_count": memory_count
                    })

                return users

        except Exception as e:
            logger.error(f"Error listing users: {e}")
            raise

    def create_user(self, name: str) -> str:
        """Create a new user and return user_id"""
        try:
            # Generate a user ID based on name and timestamp
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            user_id = f"user_{name.lower().replace(' ', '_')}_{timestamp}"

            # Create initial system message for the user
            with get_db() as session:
                system_message = Message(user_id=user_id,
                                         thread_id=f"thread_init_{timestamp}",
                                         type="system",
                                         content="User created",
                                         timestamp=datetime.now(timezone.utc),
                                         extra_metadata={"name": name})
                session.add(system_message)
                session.commit()

            logger.info(f"Created new user: {user_id}")
            return user_id

        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a user"""
        try:
            with get_db() as session:
                # Get basic user info from latest message
                user = (session.query(Message).filter_by(
                    user_id=user_id).order_by(
                        Message.timestamp.desc()).first())

                if not user:
                    return None

                # Get statistics
                message_count = (session.query(Message).filter_by(
                    user_id=user_id).count())

                memory_count = (session.query(MemoryModelDB).filter_by(
                    user_id=user_id).count())

                fact_count = (session.query(Fact).filter_by(
                    user_id=user_id).count())

                return {
                    "user_id": user_id,
                    "last_active": user.timestamp,
                    "message_count": message_count,
                    "memory_count": memory_count,
                    "fact_count": fact_count,
                    "metadata": user.extra_metadata
                }

        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            raise

    def get_latest_thread(self, user_id: str) -> Optional[str]:
        """Get the most recent thread ID for a user"""
        try:
            with get_db() as session:
                latest_message = (session.query(Message).filter_by(
                    user_id=user_id).order_by(
                        Message.timestamp.desc()).first())
                return latest_message.thread_id if latest_message else None
        except Exception as e:
            logger.error(
                f"Error getting latest thread for user {user_id}: {e}")
            return None

    def create_or_get_thread(self, user_id: str) -> str:
        """Get existing thread or create new one if needed"""
        latest_thread = self.get_latest_thread(user_id)
        if latest_thread:
            # Check if the latest thread is still active (e.g., less than 24 hours old)
            with get_db() as session:
                latest_message = (session.query(Message).filter_by(
                    user_id=user_id, thread_id=latest_thread).order_by(
                        Message.timestamp.desc()).first())
                if latest_message:
                    time_diff = datetime.now(
                        timezone.utc) - latest_message.timestamp
                    if time_diff.total_seconds(
                    ) < 24 * 3600:  # Less than 24 hours
                        return latest_thread

        # Create new thread if no active thread exists
        return f"thread_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    def get_user_threads(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all threads for a user with their last activity"""
        try:
            with get_db() as session:
                threads = (session.query(
                    Message.thread_id,
                    func.max(Message.timestamp).label('last_activity'),
                    func.count(Message.id).label('message_count')).filter_by(
                        user_id=user_id).group_by(Message.thread_id).order_by(
                            func.max(Message.timestamp).desc()).all())
                return [{
                    "thread_id": thread.thread_id,
                    "last_activity": thread.last_activity,
                    "message_count": thread.message_count
                } for thread in threads]
        except Exception as e:
            logger.error(f"Error getting threads for user {user_id}: {e}")
            return []

    def validate_thread(self, user_id: str, thread_id: str) -> bool:
        """Validate that a thread exists and belongs to the specified user"""
        try:
            with get_db() as session:
                return (session.query(Message).filter_by(
                    user_id=user_id, thread_id=thread_id).first()) is not None
        except Exception as e:
            logger.error(f"Error validating thread: {e}")
            return False


class CLI:
    """Enhanced CLI interface for chat interactions"""

    def __init__(self):
        self.manager = None
        self.debug_mode = False

    def clear_screen(self):
        """Clear screen with cross-platform support"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def handle_command(self, command: str) -> bool:
        """Handle CLI commands with enhanced error handling"""
        try:
            parts = command.strip().split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            commands = {
                '/help': lambda: display_help(),
                '/status': lambda: display_status(self.manager),
                '/clear': lambda: self.clear_screen(),
                '/quit': lambda: self.handle_quit(),
                '/reset': lambda: self.handle_reset(),
                '/debug': lambda: self.toggle_debug(),
                '/memories': lambda: self.show_memories(arg),
                '/search': lambda: self.search_memories(arg),
                '/facts': lambda: self.search_facts(arg),
                '/summary': lambda: print(history.get_thread_summary()),
            }

            if cmd in commands:
                commands[cmd]()
                return True
            return False

        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")
            print(f"❌ Error executing command: {str(e)}")
            return True

    def handle_quit(self) -> bool:
        """Handle quit command with cleanup"""
        try:
            if self.manager:
                self.manager.end_conversation()
            print("\n👋 Thanks for chatting! Your conversation has been saved.")
            return True
        except Exception as e:
            logger.error(f"Error during quit: {e}")
            print("❌ Error during quit, but exiting anyway")
            return True

    def handle_reset(self):
        """Reset conversation state"""
        try:
            if self.manager:
                self.manager.reset_conversation()
            print("🔄 Conversation has been reset")
        except Exception as e:
            logger.error(f"Error resetting conversation: {e}")
            print("❌ Error resetting conversation")

    def toggle_debug(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        print(f"🔧 Debug mode: {'enabled' if self.debug_mode else 'disabled'}")

    def show_memories(self, arg: str):
        """Display user memories with optional importance filter"""
        try:
            if not self.manager or not self.manager.current_user_id:
                print("❌ No active conversation")
                return

            importance = int(arg) if arg.isdigit() else 0
            memories = memory_store.get_memories(self.manager.current_user_id,
                                                 min_importance=importance)

            if not memories:
                print("No memories found")
                return

            print("\n📚 User Memories:")
            for mem in memories:
                print(f"[{mem['importance']}★] {mem['content']}")

        except Exception as e:
            logger.error(f"Error displaying memories: {e}")
            print("❌ Error retrieving memories")

    def search_memories(self, query: str):
        """Search user memories"""
        if not query:
            print("❗ Please provide a search term")
            return

        try:
            if not self.manager or not self.manager.current_user_id:
                print("❌ No active conversation")
                return

            results = memory_store.search_memories(
                self.manager.current_user_id, query)

            if not results:
                print(f"No memories found matching '{query}'")
                return

            print(f"\n🔍 Found {len(results)} memories:")
            for mem in results:
                print(f"[{mem['importance']}★] {mem['content']}")

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            print("❌ Error searching memories")

    def search_facts(self, query: str):
        """Search verified facts"""
        if not query:
            print("❗ Please provide a search term")
            return

        try:
            results = memory_store.search_facts(query)
            if not results:
                print(f"No facts found matching '{query}'")
                return

            print("\n📚 Verified Facts:")
            for fact in results:
                print(fact)

        except Exception as e:
            logger.error(f"Error searching facts: {e}")
            print("❌ Error searching facts")

    def run(self):
        """Main chat execution loop with enhanced error handling"""
        try:
            self.clear_screen()
            print("""
🤖 AI Chat Session
================
Type /help for available commands
""")

            while True:
                try:
                    # Get user input
                    user_input = input("\n👤 You: ").strip()

                    # Skip empty input
                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith('/'):
                        if self.handle_command(user_input):
                            if user_input.lower() == '/quit':
                                break
                            continue

                    # Process message
                    print("\n🤖 Assistant: ", end='', flush=True)
                    response = self.manager.process_message(user_input)

                    # Handle potential errors in response
                    if response.startswith("❌ Error:"):
                        logger.error(f"Error in response: {response}")
                        print("\n⚠️ Something went wrong. Please try again.")
                        continue

                    print(response)

                except KeyboardInterrupt:
                    print("\n⚠️ Use /quit to end the conversation properly")
                except Exception as e:
                    logger.error(f"Error in conversation loop: {e}")
                    print("\n❌ An error occurred. Please try again.")

                    # Reset conversation if too many errors
                    if self.manager.error_count >= self.manager.max_errors:
                        self.handle_reset()

        except Exception as e:
            logger.error(f"Fatal error in CLI: {e}")
            print("\n❌ A fatal error occurred. The program will exit.")
            return 1

        return 0


class UserInterface:
    """Enhanced CLI interface with improved user management"""

    def __init__(self):
        self.user_manager = UserManager()
        self.conversation_manager = None
        self.current_user_id = None
        self.debug_mode = False

    def clear_screen(self):
        """Clear screen with cross-platform support"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def display_main_menu(self):
        """Display main menu options"""
        print("""
🤖 AI Assistant - Main Menu
=========================
1. Create New User
2. Select Existing User
3. List All Users
4. Exit
""")

    def create_new_user(self):
        """Handle new user creation"""
        try:
            print("\n📝 Create New User")
            print("=================")
            name = input("Enter user name: ").strip()

            if not name:
                print("❌ Name cannot be empty")
                return

            user_id = self.user_manager.create_user(name)
            print(f"\n✅ User created successfully!")
            print(f"User ID: {user_id}")

            if input("\nStart conversation with this user? (y/n): ").lower(
            ).strip() == 'y':
                self.start_conversation(user_id)

        except Exception as e:
            logger.error(f"Error creating user: {e}")
            print(f"❌ Error creating user: {str(e)}")

    def list_users(self):
        """Display list of existing users"""
        try:
            users = self.user_manager.list_users()

            if not users:
                print("\n📝 No users found")
                return

            print("\n📝 Existing Users")
            print("===============")

            for user in users:
                last_active = user['last_active'].strftime(
                    '%Y-%m-%d %H:%M:%S') if user['last_active'] else "Never"
                print(f"\nUser ID: {user['user_id']}")
                print(f"Last Active: {last_active}")
                print(f"Stored Memories: {user['memory_count']}")

        except Exception as e:
            logger.error(f"Error listing users: {e}")
            print(f"❌ Error listing users: {str(e)}")

    def select_user(self):
        """Handle user selection with thread management"""
        try:
            self.list_users()

            user_id = input(
                "\nEnter User ID to select (or press Enter to cancel): "
            ).strip()

            if not user_id:
                return

            user_info = self.user_manager.get_user_info(user_id)

            if not user_info:
                print("❌ User not found")
                return

            print("\n📊 User Information")
            print("=================")
            print(f"User ID: {user_info['user_id']}")
            print(
                f"Last Active: {user_info['last_active'].strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(f"Message Count: {user_info['message_count']}")
            print(f"Memory Count: {user_info['memory_count']}")
            print(f"Fact Count: {user_info['fact_count']}")

            # Show available threads
            threads = self.user_manager.get_user_threads(
                user_id
            )  # Assuming the method is correctly defined in UserManager class
            if threads:
                print("\n🧵 Available Threads:")
                for idx, thread in enumerate(threads, 1):
                    print(f"{idx}. Thread: {thread['thread_id']}")
                    print(
                        f"   Last Activity: {thread['last_activity'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    print(f"   Messages: {thread['message_count']}")

                choice = input(
                    "\nSelect thread number or press Enter for new thread: "
                ).strip()
                selected_thread = threads[
                    int(choice) - 1]['thread_id'] if choice.isdigit(
                    ) and 0 < int(choice) <= len(threads) else None
            else:
                selected_thread = None

            if input("\nStart conversation with this user? (y/n): ").lower(
            ).strip() == 'y':
                self.start_conversation(user_id, selected_thread)

        except Exception as e:
            logger.error(f"Error selecting user: {e}")
            print(f"❌ Error selecting user: {str(e)}")

    def start_conversation(self,
                           user_id: str,
                           thread_id: Optional[str] = None):
        """Start conversation with proper thread handling"""
        try:
            self.current_user_id = user_id
            self.conversation_manager = ConversationManager()

            # Start conversation manager with selected thread
            self.conversation_manager.start_conversation(user_id=user_id,
                                                         thread_id=thread_id)

            # Start CLI with selected user
            cli = CLI()
            cli.manager = self.conversation_manager
            cli.run()

        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
            print(f"❌ Error starting conversation: {str(e)}")

    def run(self):
        """Main execution loop with enhanced error handling"""
        try:
            while True:
                self.clear_screen()
                print("""
🤖 AI Assistant v2.0
==================
Welcome to the AI Assistant!
""")
                self.display_main_menu()

                choice = input("\nEnter your choice (1-4): ").strip()

                if choice == '1':
                    self.create_new_user()
                elif choice == '2':
                    self.select_user()
                elif choice == '3':
                    self.list_users()
                    input("\nPress Enter to continue...")
                elif choice == '4':
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")
                    input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            logger.error(f"Fatal error in user interface: {e}")
            print(f"\n❌ Fatal error: {str(e)}")
            return 1

        return 0


def display_help():
    """Display comprehensive help information"""
    help_text = """
🤖 Enhanced AI Assistant Commands & Features
==========================================

Basic Commands:
--------------
/help      - Show this help message
/status    - Show current conversation status
/memories  - Display user memories
/facts     - Search saved facts
/search    - Search user information
/clear     - Clear screen
/reset     - Reset current conversation
/quit      - End conversation and exit

Memory Commands:
---------------
/memories [importance]   - Show memories (optional: filter by importance 1-5)
/search <term>          - Search user memories
/facts <query>          - Search verified facts

Conversation Commands:
--------------------
/summary   - Show conversation summary
/context   - Show current context
/debug     - Toggle debug mode

Tips:
-----
- The AI automatically saves important information
- Use natural language for normal conversation
- Start questions with '?' for explicit information seeking
- The AI maintains context across conversations
- All memories and facts are persistent

Examples:
--------
/memories 3           - Show memories with importance ≥ 3
/search hobbies      - Search memories about hobbies
/facts climate       - Search verified facts about climate

For more detailed documentation, visit: <documentation_url>
"""
    print(help_text)


def display_status(manager: ConversationManager):
    """Display current conversation status"""
    try:
        status = f"""
📊 Conversation Status
=====================
User ID: {manager.current_user_id}
Thread ID: {manager.current_thread_id}
Error Count: {manager.error_count}
Active: {'Yes' if manager.current_user_id else 'No'}

{history.get_thread_summary()}
"""
        print(status)
    except Exception as e:
        logger.error(f"Error displaying status: {e}")
        print("❌ Error retrieving status")


def check_environment():
    """Verify environment setup"""
    required_vars = {
        "ANTHROPIC_API_KEY":
        "Anthropic API key for Claude",
        "TAVILY_API_KEY":
        "Tavily API key for search",
        "DATABASE_URL":
        "Database connection string (or individual PGXXX variables)",
    }

    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            # Check for database URL components if DATABASE_URL is missing
            if var == "DATABASE_URL" and all(
                    os.getenv(f"PG{comp}") for comp in
                ["DATABASE", "HOST", "PORT", "USER", "PASSWORD"]):
                continue
            missing.append(f"- {var}: {description}")

    if missing:
        print("❌ Missing required environment variables:")
        print("\n".join(missing))
        print("\nPlease set these variables in your environment or .env file.")
        return False

    return True


# Add after the CLI class and before the main function:


def main():
    """Main entry point with enhanced error handling"""
    try:
        # Verify environment
        if not check_environment():
            return 1

        # Initialize database
        if not initialize_database():
            logger.error("Database initialization failed. Exiting.")
            return 1

        logger.info("Starting application...")

        # Create and run user interface
        ui = UserInterface()
        return ui.run()

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"\n❌ Fatal error: {str(e)}")
        print("The program will now exit.")
        return 1


if __name__ == "__main__":
    # Ensure clean shutdown
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
