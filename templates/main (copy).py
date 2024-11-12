from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent, InjectedStore
from typing import Annotated, Any, Dict, List, Union, Optional
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,        
    Column,
    String,
    Integer,
    Text,
    DateTime,  
    JSON,
    desc,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing_extensions import TypedDict
import json
import os
import sys

# SQLAlchemy setup
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Construct DATABASE_URL from individual components if not provided
    PGDATABASE = os.getenv("PGDATABASE")
    PGHOST = os.getenv("PGHOST")
    PGPORT = os.getenv("PGPORT")
    PGUSER = os.getenv("PGUSER")
    PGPASSWORD = os.getenv("PGPASSWORD")

    if not all([PGDATABASE, PGHOST, PGPORT, PGUSER, PGPASSWORD]):
        print(
            "❌ Error: Missing required environment variables for PostgreSQL connection."
        )
        sys.exit(1)

    DATABASE_URL = (
        f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    )

# Initialize the database engine with connection pooling and pre-ping to handle stale connections
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(bind=engine)

# Base class for declarative models
Base = declarative_base()


# SQLAlchemy Models
class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    thread_id = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False)  # 'system', 'human', 'ai'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    extra_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata'

    def to_message_object(self) -> Union[SystemMessage, HumanMessage, AIMessage]:
        """
        Convert the database Message instance to a LangChain Message object,
        including the timestamp in additional_kwargs.
        """
        kwargs = {"additional_kwargs": {"timestamp": self.timestamp.isoformat()}}
        if self.type == "system":
            return SystemMessage(content=self.content, **kwargs)
        elif self.type == "human":
            return HumanMessage(content=self.content, **kwargs)
        elif self.type == "ai":
            return AIMessage(content=self.content, **kwargs)
        else:
            raise ValueError(f"Unknown message type: {self.type}")


class MemoryModelDB(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    content = Column(Text, nullable=False)
    importance = Column(Integer, default=1, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    extra_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata'


# Create tables if they don't exist
Base.metadata.create_all(bind=engine)


# Custom types for better type hinting
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


# Initialize a session
def get_db_session():
    return SessionLocal()


class MessageHistory:
    """
    Enhanced Message History handler with proper message saving and retrieval.
    Organizes conversations by user_id and thread_id.
    """

    def __init__(self):
        self.messages: List[Union[SystemMessage, HumanMessage, AIMessage]] = []
        self.current_user_id: Optional[str] = None
        self.current_thread_id: Optional[str] = None

    def load_thread(
        self, user_id: str, thread_id: str
    ) -> List[Union[SystemMessage, HumanMessage, AIMessage]]:
        """
        Load a specific conversation thread for a user.
        Creates new if doesn't exist.
        """
        self.current_user_id = user_id
        self.current_thread_id = thread_id
        self.messages = []

        session = get_db_session()
        try:
            db_messages = (
                session.query(Message)
                .filter_by(user_id=user_id, thread_id=thread_id)
                .order_by(Message.timestamp)
                .all()
            )
            if db_messages:
                for db_msg in db_messages:
                    self.messages.append(db_msg.to_message_object())
            else:
                # Initialize with default system message if new thread
                system_message = SystemMessage(
                    content="""You are a helpful AI assistant. Remember important information about users. 
Use tools effectively to search for information and save key user details. 
Maintain context and provide personalized responses based on user history.""",
                    additional_kwargs={"timestamp": datetime.utcnow().isoformat()},
                )
                self.add_message(system_message, save_to_db=True)
        except Exception as e:
            print(f"Error loading thread: {e}")
        finally:
            session.close()

        return self.messages

    def add_message(
        self, message: Union[SystemMessage, HumanMessage, AIMessage], save_to_db: bool = True
    ) -> None:
        """Add a new message to current thread with enhanced processing"""
        if not self.current_user_id or not self.current_thread_id:
            raise ValueError("No active thread")

        # Handle system messages specially
        if isinstance(message, SystemMessage):
            # Replace existing system message if any
            self.messages = [m for m in self.messages if not isinstance(m, SystemMessage)]
            self.messages.insert(0, message)
        else:
            # Add non-system messages
            self.messages.append(message)

            # Process AI messages for potential memory storage
            if isinstance(message, AIMessage):
                self._process_ai_memory(message)
                

        if save_to_db:
            self.save_message_to_db(message)

    def save_message_to_db(
        self, message: Union[SystemMessage, HumanMessage, AIMessage]) -> None:
        """Save a single message with detailed error handling"""
        session = get_db_session()
        try:
            # Determine message type
            msg_type = type(message).__name__.lower().replace("message", "")
            logger.debug(f"Prepared message type: {msg_type}")

            # Extract and format timestamp
            timestamp_str = message.additional_kwargs.get("timestamp")
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.utcnow()

            db_msg = Message(
                user_id=self.current_user_id,
                thread_id=self.current_thread_id,
                type=msg_type,
                content=message.content if isinstance(message.content, str) else json.dumps(message.content),
                timestamp=timestamp,
                extra_metadata=message.additional_kwargs if message.additional_kwargs else {},
            )

            logger.info(f"Saving message to DB: {db_msg.content[:30]}... (type: {db_msg.type})")
            session.add(db_msg)
            session.commit()
            logger.info("✅ Message successfully saved to DB.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error during database save: {e} | Message content: {message.content[:30]}...")
        finally:
            session.close()

    def _process_ai_memory(self, message: AIMessage) -> None:
        """Process AI messages for memory saving triggers"""
        # Keywords that might indicate important information to remember
        memory_triggers = [
            "i'll remember that",
            "important to note that",
            "i should remember that",
            "noted that",
            "you mentioned that",
        ]

        content_lower = message.content.lower() if isinstance(message.content, str) else ""
        if any(trigger in content_lower for trigger in memory_triggers):
            memory_store.save_memory(
                self.current_user_id,
                message.content if isinstance(message.content, str) else json.dumps(message.content),
                {"type": "auto_saved", "source": "ai_message"},
                importance=3,  # Default importance for auto-saved memories
            )

    def get_messages(self) -> List[Union[SystemMessage, HumanMessage, AIMessage]]:
        """Get all messages from current thread with system message validation"""
        if not self.messages or not any(isinstance(m, SystemMessage) for m in self.messages):
            system_message = SystemMessage(
                content="""You are a helpful AI assistant. Remember important information about users. 
Use tools effectively to search for information and save key user details. 
Maintain context and provide personalized responses based on user history.""",
                additional_kwargs={"timestamp": datetime.utcnow().isoformat()},
            )
            self.add_message(system_message, save_to_db=True)
        return self.messages

    def get_thread_summary(self) -> str:
        """Generate a summary of the current thread"""
        if not self.messages:
            return "No messages in current thread."

        msg_count = len(self.messages)
        last_msg = self.messages[-1]
        last_msg_time = last_msg.additional_kwargs.get("timestamp", datetime.utcnow().isoformat())

        return f"Thread contains {msg_count} messages. Last activity: {last_msg_time}"


class UserMemoryStore:
    """
    Enhanced storage system for user memories with metadata and retrieval functions
    """

    def save_memory(
        self,
        user_id: str,
        memory: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: int = 1,
    ) -> None:
        """
        Save a memory with metadata and timestamp
        """
        session = get_db_session()
        try:
            db_memory = MemoryModelDB(
                user_id=user_id,
                content=memory,
                importance=importance,
                timestamp=datetime.utcnow(),
                extra_metadata=metadata or {},
            )
            session.add(db_memory)
            session.commit()
            print(f"✅ Saved memory to DB: {memory}")
        except Exception as e:
            session.rollback()
            print(f"Error saving memory: {e}")
        finally:
            session.close()

    def get_memories(self, user_id: str) -> List[MemoryEntry]:
        """
        Retrieve all memories for a user
        """
        session = get_db_session()
        memories: List[MemoryEntry] = []
        try:
            db_memories = (
                session.query(MemoryModelDB)
                .filter_by(user_id=user_id)
                .order_by(MemoryModelDB.timestamp)
                .all()
            )
            for mem in db_memories:
                memories.append(
                    {
                        "content": mem.content,
                        "timestamp": mem.timestamp.isoformat(),
                        "importance": mem.importance,
                        "metadata": mem.extra_metadata,
                    }
                )
            print(f"📥 Retrieved {len(memories)} memories from DB for user {user_id}")
        except Exception as e:
            print(f"Error retrieving memories: {e}")
        finally:
            session.close()
        return memories

    def get_recent_memories(
        self, user_id: str, limit: int = 5
    ) -> List[MemoryEntry]:
        """
        Get most recent memories for a user
        """
        session = get_db_session()
        memories: List[MemoryEntry] = []
        try:
            db_memories = (
                session.query(MemoryModelDB)
                .filter_by(user_id=user_id)
                .order_by(MemoryModelDB.timestamp.desc())
                .limit(limit)
                .all()
            )
            for mem in db_memories:
                memories.append(
                    {
                        "content": mem.content,
                        "timestamp": mem.timestamp.isoformat(),
                        "importance": mem.importance,
                        "metadata": mem.extra_metadata,
                    }
                )
            print(f"📥 Retrieved {len(memories)} recent memories from DB for user {user_id}")
        except Exception as e:
            print(f"Error retrieving recent memories: {e}")
        finally:
            session.close()
        return memories

    def search_memories(
        self, user_id: str, query: str
    ) -> List[MemoryEntry]:
        """
        Search user memories for specific content
        """
        session = get_db_session()
        results: List[MemoryEntry] = []
        try:
            db_memories = (
                session.query(MemoryModelDB)
                .filter(MemoryModelDB.user_id == user_id)
                .filter(MemoryModelDB.content.ilike(f"%{query}%"))
                .order_by(MemoryModelDB.timestamp.desc())
                .all()
            )
            for mem in db_memories:
                results.append(
                    {
                        "content": mem.content,
                        "timestamp": mem.timestamp.isoformat(),
                        "importance": mem.importance,
                        "metadata": mem.extra_metadata,
                    }
                )
            print(f"🔍 Found {len(results)} memories matching '{query}' for user {user_id}")
        except Exception as e:
            print(f"Error searching memories: {e}")
        finally:
            session.close()
        return results


# Initialize storage systems
history = MessageHistory()
memory_store = UserMemoryStore()
store = InMemoryStore()


class MemoryModel(BaseModel):
    """Schema for memory storage"""
    content: str = Field(..., description="The memory content to save")
    importance: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Importance level of the memory (1-5)",
    )


@tool
def save_memory(
    memory: str,
    importance: int = 1,
    *,
    config: RunnableConfig,
    store: Annotated[Any, InjectedStore()],  # Included for compatibility
) -> str:
    """
    Save important information about the user with importance level.
    Args:
        memory: The information to save
        importance: Importance level (1-5), default is 1
    Returns:
        Confirmation message
    """
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "❌ Error: No user ID provided"

    metadata = {
        "importance": importance,
        "source": "explicit_save",
        "tool_call": True,
    }

    memory_store.save_memory(user_id, memory, metadata, importance=importance)
    return f"✅ Saved memory with importance level {importance}: {memory}"


@tool
def search_user_info(
    query: str,
    *,
    config: RunnableConfig,
    store: Annotated[Any, InjectedStore()],
) -> str:
    """
    Search through user's stored memories for specific information.
    Args:
        query: The information to search for
    Returns:
        Relevant memories if found
    """
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "❌ Error: No user ID provided"

    results = memory_store.search_memories(user_id, query)
    if not results:
        return f"No memories found matching '{query}'"

    response = f"🔍 Found {len(results)} relevant memories:\n"
    for memory in results:
        importance = memory.get("importance", 1)
        response += f"- {memory['content']} (Importance: {importance}★)\n"
    return response


def get_user_memories(user_id: str) -> str:
    """
    Format user memories for model context
    """
    memories = memory_store.get_memories(user_id)
    if not memories:
        return "No previous information about this user."

    # Sort memories by importance
    sorted_memories = sorted(
        memories, key=lambda x: x.get("importance", 1), reverse=True
    )

    memory_text = "📚 Previous memories about this user:\n"
    for memory in sorted_memories:
        importance = memory.get("importance", 1)
        timestamp = datetime.fromisoformat(memory["timestamp"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        memory_text += f"- [{importance}★] {memory['content']} (Saved: {timestamp})\n"

    return memory_text


def prepare_model_inputs(
    state: Dict,
    config: RunnableConfig,
    # store: InjectedStore
) -> List[Union[SystemMessage, HumanMessage, AIMessage]]:
    """
    Prepare messages for the model including user memories and history
    """
    # Get user memories
    user_id = config.get("configurable", {}).get("user_id")
    memories = get_user_memories(user_id)

    # Create system message with memories and instructions
    system_message = SystemMessage(
        content=f"""You are a helpful AI assistant with access to user memories.

IMPORTANT USER CONTEXT:
{memories}

INSTRUCTIONS:
1. Use the above user context to provide personalized responses
2. Maintain consistency with known user information
3. Save new important information using the save_memory tool
4. Use the search_user_info tool to find specific user details
5. When saving memories, use importance levels:
   - 5: Critical personal information (name, location, major life events)
   - 4: Preferences and important facts
   - 3: General interests and regular activities
   - 2: Casual mentions and minor details
   - 1: Temporary or contextual information

Always consider the user's history when responding and maintain a friendly, consistent interaction based on what you know about them.""",
        additional_kwargs={"timestamp": datetime.utcnow().isoformat()},
    )

    # Get conversation history
    messages = history.get_messages()

    # Replace system message with our enhanced version
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = system_message
    else:
        messages.insert(0, system_message)

    # Add current messages (excluding system messages)
    current_messages = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    messages.extend(current_messages)

    return messages


# Create the agent with enhanced configuration
def create_agent():
    """
    Create and configure the agent with all necessary tools and settings
    """
    # Initialize core components
    memory = MemorySaver()
    model = ChatAnthropic(
        model_name="claude-3-haiku-20240307",
        temperature=0.7,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Initialize tools
    search = TavilySearchResults(
        max_results=3,
        api_key=os.getenv("TAVILY_API_KEY"),
    )

    tools = [
        search,
        save_memory,
        search_user_info,
    ]

    # Create the agent with all components
    return create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        store=store,
        state_modifier=prepare_model_inputs,
        debug=True,  # Enable debug mode for better error tracking
    )


def process_response(chunk: Dict):
    """
    Enhanced response processor with better formatting and error handling
    """
    try:
        if isinstance(chunk, dict) and "messages" in chunk:
            for message in chunk["messages"]:
                # Print message with clear formatting
                print(f"\n{'=' * 60}")

                if isinstance(message, AIMessage):
                    print("🤖 AI:", message.content)
                elif isinstance(message, HumanMessage):
                    print("👤 Human:", message.content)
                else:
                    continue  # Skip system messages in output

                print(f"{'=' * 60}\n")

                # Save non-system messages to history
                if not isinstance(message, SystemMessage):
                    history.add_message(message)

    except Exception as e:
        print(f"Error processing response: {str(e)}")
        print("Raw chunk:", chunk)


class ConversationManager:
    """
    Manages the conversation flow and state
    """

    def __init__(self):
        self.agent = create_agent()
        self.config = {}

    def start_conversation(self, user_id: str = None, thread_id: str = None):
        """
        Start or resume a conversation
        """
        # Generate IDs if not provided
        user_id = user_id or input("Enter user ID (or press Enter for default 'user_1'): ").strip() or "user_1"
        thread_id = thread_id or f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Setup configuration
        self.config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            }
        }

        # Load conversation history
        history.load_thread(user_id, thread_id)

        # Print session information
        print("\n" + "=" * 60)
        print(f"🆔 User ID: {user_id}")
        print(f"📝 Thread ID: {thread_id}")
        print("=" * 60)

        # Print user memories
        print("\n📚 User Information:")
        print(get_user_memories(user_id))
        print("=" * 60 + "\n")

        return self.config

    def process_message(self, user_input: str) -> str:
        try:
            user_message = HumanMessage(
                content=user_input,
                additional_kwargs={"timestamp": datetime.utcnow().isoformat()}
            )
            history.add_message(user_message)  # Add user message

            # Generate AI response
            response_parts = []
            for chunk in self.agent.stream(
                {
                    "messages": [user_message],
                },
                self.config,
            ):
                response = process_response(chunk)
                if response:
                    response_parts.append(response)

            if response_parts:
                # Combine parts into final response content
                final_response = ''.join(response_parts)

                # Log the AI response for debugging
                logger.info(f"Generated AI Response: {final_response}")

                # Construct AI message
                ai_message = AIMessage(
                    content=final_response,
                    additional_kwargs={"timestamp": datetime.utcnow().isoformat()}
                )

                # Add AI message to history and ensure saving to DB
                history.add_message(ai_message, save_to_db=True)

                return final_response
            else:
                return "No valid response generated by AI."

        except Exception as e:
            error_message = f"❌ Error processing message: {str(e)}"
            logger.error(error_message)
            return error_message

def display_help():
    """
    Display available commands and usage information
    """
    print(
        """
Available Commands:
------------------
/help      - Display this help message
/memories  - Show all memories for current user
/search    - Search user memories (follow with search term)
/clear     - Clear the screen
/quit      - End conversation

Special Features:
-----------------
- The AI automatically saves important information about you
- Use natural language to interact
- The AI maintains context across conversations
- All memories and conversations are persistent
"""
    )


def main():
    """
    Main execution loop with enhanced error handling and commands
    """
    try:
        # Clear screen for better UX
        os.system("cls" if os.name == "nt" else "clear")

        print(
            """
🤖 Enhanced AI Assistant with Memory
==================================
Type /help for available commands
"""
        )

        # Initialize conversation
        manager = ConversationManager()
        config = manager.start_conversation()

        while True:
            try:
                # Get user input
                user_input = input("\n👤 Enter your message (or /help for commands): ").strip()

                # Handle commands
                if user_input.lower() == "/quit":
                    break
                elif user_input.lower() == "/help":
                    display_help()
                    continue
                elif user_input.lower() == "/memories":
                    print("\n" + get_user_memories(config["configurable"]["user_id"]))
                    continue
                elif user_input.lower().startswith("/search "):
                    query = user_input[8:].strip()
                    if not query:
                        print("❗ Please provide a search term after /search.")
                        continue
                    print(
                        "\n" + search_user_info(
                            query,
                            config=config,
                            store=store,
                        )
                    )
                    continue
                elif user_input.lower() == "/clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue
                elif not user_input:
                    continue

                # Process regular message
                manager.process_message(user_input)

            except KeyboardInterrupt:
                print("\n⚠️ Use /quit to end the conversation properly")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("Please try again or use /quit to exit")

        # Clean up
        manager.end_conversation()

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print("The program will now exit.")
        return 1

    return 0


if __name__ == "__main__":
    # Ensure environment variables are set
    required_vars = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables before running the program.")
        sys.exit(1)

    sys.exit(main())