import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Float, ForeignKey, Text, or_, and_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB

# LangChain imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# Model Definitions
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    threads = relationship("Thread", back_populates="user")
    memories = relationship("UserMemory", back_populates="user")


class UserMemory(Base):
    __tablename__ = "user_memories"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    category = Column(String)
    attribute = Column(String)
    value = Column(Text)
    confidence = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="memories")

    @classmethod
    def create_or_update(cls, session, user_id: str, category: str,
                         attribute: str, value: str, confidence: float):
        try:
            existing = session.query(cls).filter_by(
                user_id=user_id, category=category,
                attribute=attribute).first()

            if existing:
                if confidence > existing.confidence:
                    existing.value = value
                    existing.confidence = confidence
                    existing.last_updated = datetime.utcnow()
            else:
                memory = cls(user_id=user_id,
                             category=category,
                             attribute=attribute,
                             value=value,
                             confidence=confidence)
                session.add(memory)

            session.commit()
            logger.info(
                f"Memory created/updated for user {user_id}: {category}.{attribute}"
            )
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error in create_or_update memory: {str(e)}")
            return False


class FactDatabase(Base):
    __tablename__ = "facts"
    id = Column(Integer, primary_key=True)
    category = Column(String)
    title = Column(String)
    content = Column(Text)
    keywords = Column(JSONB)
    confidence = Column(Float)
    source_urls = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_verified = Column(DateTime, default=datetime.utcnow)
    expiry_date = Column(DateTime, nullable=True)

    @classmethod
    def search(cls, session, query: str, min_confidence: float = 0.7):
        try:
            keywords = set(word.lower() for word in query.split())
            return session.query(cls).filter(
                and_(
                    cls.confidence >= min_confidence,
                    or_(*[
                        cls.keywords.contains([keyword])
                        for keyword in keywords
                    ]),
                    or_(cls.expiry_date.is_(None), cls.expiry_date
                        > datetime.utcnow()))).order_by(
                            cls.last_verified.desc()).limit(3).all()
        except Exception as e:
            logger.error(f"Error in fact search: {str(e)}")
            return []

    @classmethod
    def create_fact(cls,
                    session,
                    category: str,
                    title: str,
                    content: str,
                    keywords: List[str],
                    confidence: float,
                    sources: List[str],
                    expiry_days: Optional[int] = None):
        try:
            fact = cls(category=category,
                       title=title,
                       content=content,
                       keywords=keywords,
                       confidence=confidence,
                       source_urls=sources,
                       expiry_date=datetime.utcnow() +
                       timedelta(days=expiry_days) if expiry_days else None)
            session.add(fact)
            session.commit()
            logger.info(f"New fact created: {title}")
            return fact
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating fact: {str(e)}")
            return None


class Thread(Base):
    __tablename__ = "threads"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    summary = Column(Text)
    user = relationship("User", back_populates="threads")
    messages = relationship("Message", back_populates="thread")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    thread_id = Column(String, ForeignKey("threads.id"))
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    thread = relationship("Thread", back_populates="messages")


# Create tables
Base.metadata.create_all(engine)


# Memory Extraction Class
class MemoryExtractor:
    MEMORY_CATEGORIES = {
        "PERSONAL":
        ["name", "age", "occupation", "location", "interests", "family"],
        "PREFERENCES": ["likes", "dislikes", "favorites"],
        "SENTIMENT": ["mood", "attitude", "emotions"],
        "CONTEXT": ["current_topic", "recent_events"]
    }

    def __init__(self, model: ChatMistralAI):
        self.model = model

    async def extract_memories(self, message: str) -> List[Dict[str, Any]]:
        try:
            extraction_prompt = f"""Task: Extract user information from the message below.
Categories and attributes allowed:
{json.dumps(self.MEMORY_CATEGORIES, indent=2)}
Message: "{message}"
Instructions:
1. Identify any personal information, preferences, sentiments, or contextual details
2. For each piece of information, specify:
   - category (from the allowed categories)
   - attribute (from the allowed attributes)
   - value (the actual information)
   - confidence (0.0-1.0)
3. Return in this exact JSON format:
   [
     {{
       "category": "CATEGORY_NAME",
       "attribute": "attribute_name",
       "value": "extracted_value",
       "confidence": confidence_score
     }}
   ]
4. If no relevant information found, return: []
Ensure the response is valid JSON."""

            messages = [
                SystemMessage(
                    content=
                    "You are a precise information extraction system. Only output valid JSON."
                ),
                HumanMessage(content=extraction_prompt)
            ]

            response = await self.model.ainvoke(messages)

            try:
                content = response.content
                start_idx = content.find("[")
                end_idx = content.rfind("]") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    memories = json.loads(json_str)
                else:
                    return []

                valid_memories = []
                for memory in memories:
                    if self._validate_memory(memory):
                        valid_memories.append(memory)

                logger.info(f"Extracted {len(valid_memories)} valid memories")
                return valid_memories

            except json.JSONDecodeError as e:
                logger.error(
                    f"JSON parsing error in memory extraction: {str(e)}")
                return []

        except Exception as e:
            logger.error(f"Memory extraction error: {str(e)}")
            return []

    def _validate_memory(self, memory: Dict) -> bool:
        try:
            required_keys = {"category", "attribute", "value", "confidence"}
            if not all(key in memory for key in required_keys):
                return False

            if memory["category"] not in self.MEMORY_CATEGORIES:
                return False
            if memory["attribute"] not in self.MEMORY_CATEGORIES[
                    memory["category"]]:
                return False

            if not isinstance(
                    memory["confidence"],
                (int, float)) or not 0 <= memory["confidence"] <= 1:
                return False

            if not isinstance(memory["value"],
                              str) or not memory["value"].strip():
                return False

            return True
        except Exception:
            return False


# Fact Extraction Class (No changes needed)
class FactExtractor:

    def __init__(self, model: ChatMistralAI):
        self.model = model

    async def extract_facts(self, search_results: List[Dict],
                            query: str) -> Optional[Dict]:
        try:
            fact_prompt = f"""Task: Analyze these search results and extract verified factual information.
Query: "{query}"
Search Results:
{json.dumps(search_results, indent=2)}
Instructions:
1. Identify the most relevant and verifiable fact from the search results
2. Create a fact entry with these exact fields:
   - category: main topic area
   - title: brief, descriptive title
   - content: complete, verified information
   - keywords: list of relevant search terms
   - confidence: 0.0-1.0 score of certainty
   - expiry_days: number of days until fact should be reverified (null for permanent facts)
3. Return in this exact JSON format:
   {{
     "category": "category_name",
     "title": "brief_title",
     "content": "verified_content",
     "keywords": ["keyword1", "keyword2"],
     "confidence": confidence_score,
     "expiry_days": number_or_null
   }}
4. If no clear facts found, return: null
Ensure the response is valid JSON. Only include verified information."""

            messages = [
                SystemMessage(
                    content=
                    "You are a precise fact extraction system. Only output valid JSON."
                ),
                HumanMessage(content=fact_prompt)
            ]

            response = await self.model.ainvoke(messages)

            try:
                content = response.content
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    fact = json.loads(json_str)
                else:
                    return None

                if fact and self._validate_fact(fact):
                    logger.info(f"Extracted valid fact: {fact.get('title')}")
                    return fact
                return None

            except json.JSONDecodeError as e:
                logger.error(
                    f"JSON parsing error in fact extraction: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Fact extraction error: {str(e)}")
            return None

    def _validate_fact(self, fact: Dict) -> bool:
        try:
            required_keys = {
                "category", "title", "content", "keywords", "confidence"
            }
            if not all(key in fact for key in required_keys):
                return False

            if not isinstance(
                    fact["confidence"],
                (int, float)) or not 0 <= fact["confidence"] <= 1:
                return False

            if not isinstance(fact["keywords"], list) or not all(
                    isinstance(k, str) for k in fact["keywords"]):
                return False

            if not all(
                    isinstance(fact[k], str) and fact[k].strip()
                    for k in ["category", "title", "content"]):
                return False

            if "expiry_days" in fact and fact["expiry_days"] is not None:
                if not isinstance(fact["expiry_days"],
                                  int) or fact["expiry_days"] < 0:
                    return False

            return True
        except Exception:
            return False


# Enhanced Agent Class
class EnhancedAgent:

    def __init__(self):
        self.model = ChatMistralAI(
            model="mistral-large-latest",
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.3,
        )
        self.session = SessionLocal()
        self.tools = [TavilySearchResults(max_results=3)]
        self.memory_extractor = MemoryExtractor(self.model)
        self.fact_extractor = FactExtractor(self.model)
        self.agent_executor = self._create_agent()

    def _create_agent(self):
        system_prompt = """You are an advanced AI assistant providing direct, precise, and personalized responses. 
        Incorporate user context or verified facts naturally. Always maintain a helpful, friendly tone while being efficient.
        Cite any sources or facts when used."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}")
        ])

        agent = create_openai_tools_agent(self.model, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def _should_use_rag(self, query: str) -> bool:
        time_sensitive = [
            "current", "latest", "recent", "news", "today", "now"
        ]
        fact_checking = [
            "who", "when", "where", "how many", "statistics", "data"
        ]
        general_knowledge = [
            "explain", "what is", "concept of", "meaning of", "philosophy",
            "theory", "principle"
        ]

        query_lower = query.lower()
        needs_verification = (any(term in query_lower
                                  for term in time_sensitive)
                              or any(term in query_lower
                                     for term in fact_checking))
        is_general = any(term in query_lower for term in general_knowledge)

        return needs_verification and not is_general

    async def _generate_response(self,
                                 user_input: str,
                                 thread_id: str,
                                 user_context: Dict[str, Any],
                                 facts: List[FactDatabase] = None,
                                 search_results: List[Dict] = None) -> str:
        try:
            # Build context parts from memories and facts
            context_parts = []

            if user_context:
                context_parts.append(
                    f"User Context: {json.dumps(user_context, indent=2)}")

            if facts:
                facts_info = [{
                    "title": f.title,
                    "content": f.content
                } for f in facts]
                context_parts.append(
                    f"Verified Facts: {json.dumps(facts_info, indent=2)}")

            if search_results:
                context_parts.append(
                    f"Search Results: {json.dumps(search_results, indent=2)}")

            input_text = user_input
            if context_parts:
                input_text = f"""Context Information:
                {chr(10).join(context_parts)}

                User Question: {user_input}

                Provide a clear, direct response incorporating relevant context."""

            chat_history = await self._get_chat_history(thread_id)

            response = await self.agent_executor.ainvoke({
                "input":
                input_text,
                "chat_history":
                chat_history
            })

            return response.get(
                "output",
                "I apologize, but I'm not sure how to respond to that.")

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error generating a response. Please try again."

    async def process_message(self, user_input: str, thread_id: str,
                              user_id: str) -> str:
        try:
            # Save user message
            await self._save_interaction(thread_id, "human", user_input)

            # Extract and store user memories
            memories = await self.memory_extractor.extract_memories(user_input)
            for memory in memories:
                UserMemory.create_or_update(self.session,
                                            user_id=user_id,
                                            category=memory["category"],
                                            attribute=memory["attribute"],
                                            value=memory["value"],
                                            confidence=memory["confidence"])

            # Get user context
            user_context = self._get_user_context(user_id)

            response = ""
            # Determine if RAG is needed
            if self._should_use_rag(user_input):
                # Check local fact database first
                facts = FactDatabase.search(self.session, user_input)

                if facts:
                    # Use existing facts
                    response = await self._generate_response(user_input,
                                                             thread_id,
                                                             user_context,
                                                             facts=facts)
                else:
                    # Perform web search
                    try:
                        search_results = await self.tools[0].ainvoke(user_input
                                                                     )

                        if search_results:
                            # Extract and store new facts
                            fact = await self.fact_extractor.extract_facts(
                                search_results, user_input)
                            if fact:
                                FactDatabase.create_fact(
                                    self.session,
                                    category=fact["category"],
                                    title=fact["title"],
                                    content=fact["content"],
                                    keywords=fact["keywords"],
                                    confidence=fact["confidence"],
                                    sources=[
                                        result.get("url")
                                        for result in search_results
                                        if "url" in result
                                    ],
                                    expiry_days=fact.get("expiry_days"))

                            response = await self._generate_response(
                                user_input,
                                thread_id,
                                user_context,
                                search_results=search_results)
                    except Exception as search_error:
                        logger.error(f"Search error: {str(search_error)}")
                        response = await self._generate_response(
                            user_input, thread_id, user_context)
            else:
                # Direct response without RAG
                response = await self._generate_response(
                    user_input, thread_id, user_context)

            # Save AI response
            await self._save_interaction(thread_id, "ai", response)
            return response

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

    def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        try:
            memories = (self.session.query(UserMemory).filter(
                UserMemory.user_id == user_id, UserMemory.confidence
                >= 0.7).order_by(UserMemory.last_updated.desc()).all())

            context = {}
            for memory in memories:
                if memory.category not in context:
                    context[memory.category] = {}
                context[memory.category][memory.attribute] = memory.value

            return context
        except Exception as e:
            logger.error(f"Error getting user context: {str(e)}")
            return {}

    async def _get_chat_history(self, thread_id: str, limit: int = 5) -> List:
        try:
            messages = (self.session.query(Message).filter(
                Message.thread_id == thread_id).order_by(
                    Message.created_at.desc()).limit(limit).all())

            history = []
            for msg in reversed(messages):
                if msg.role == "human":
                    history.append(HumanMessage(content=msg.content))
                elif msg.role == "ai":
                    history.append(AIMessage(content=msg.content))

            return history
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []

    async def _save_interaction(self, thread_id: str, role: str, content: str):
        try:
            # Save message
            message = Message(thread_id=thread_id, role=role, content=content)
            self.session.add(message)

            # Update thread
            thread = self.session.query(Thread).filter_by(id=thread_id).first()
            if thread:
                thread.last_active = datetime.utcnow()

            self.session.commit()
            logger.info(f"Saved {role} message in thread {thread_id}")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving interaction: {str(e)}")
            raise


# Chat Interface Class (No changes needed)
class ChatInterface:

    def __init__(self):
        self.agent = EnhancedAgent()
        self.session = SessionLocal()
        self.current_user = None
        self.current_thread = None

    def _create_user(self, name: str) -> str:
        try:
            user_id = f"user_{uuid4().hex[:8]}"
            user = User(id=user_id, name=name)
            self.session.add(user)
            self.session.commit()
            logger.info(f"Created new user: {name} (ID: {user_id})")
            return user_id
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating user: {str(e)}")
            raise

    def _create_thread(self, user_id: str) -> str:
        try:
            thread_id = f"thread_{uuid4().hex[:8]}"
            thread = Thread(id=thread_id, user_id=user_id)
            self.session.add(thread)
            self.session.commit()
            logger.info(f"Created new thread: {thread_id} for user: {user_id}")
            return thread_id
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating thread: {str(e)}")
            raise

    def _show_menu(self):
        print("\n=== Enhanced AI Assistant ===")
        print("1. Create new user")
        print("2. Select existing user")
        print("3. Start new chat")
        print("4. View user memories")
        print("5. View fact database")
        print("6. Exit")
        return input("Choose option (1-6): ")

    def _display_user_memories(self, user_id: str):
        try:
            memories = (self.session.query(UserMemory).filter_by(
                user_id=user_id).order_by(
                    UserMemory.category, UserMemory.last_updated.desc()).all())

            if memories:
                print("\n=== User Memories ===")
                current_category = None
                for memory in memories:
                    if memory.category != current_category:
                        current_category = memory.category
                        print(f"\n{current_category}:")
                    print(f"  {memory.attribute}: {memory.value}")
                    print(f"  Confidence: {memory.confidence:.2f}")
                    print(
                        f"  Last Updated: {memory.last_updated.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    print()
            else:
                print("\nNo memories found for this user.")

        except Exception as e:
            logger.error(f"Error displaying memories: {str(e)}")
            print("Error retrieving user memories.")

    def _display_facts(self):
        try:
            facts = (self.session.query(FactDatabase).order_by(
                FactDatabase.category,
                FactDatabase.last_verified.desc()).limit(10).all())

            if facts:
                print("\n=== Recent Facts ===")
                current_category = None
                for fact in facts:
                    if fact.category != current_category:
                        current_category = fact.category
                        print(f"\n{fact.category}:")
                    print(f"  Title: {fact.title}")
                    print(f"  Content: {fact.content}")
                    print(f"  Confidence: {fact.confidence:.2f}")
                    print(
                        f"  Last Verified: {fact.last_verified.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    if fact.expiry_date:
                        print(
                            f"  Expires: {fact.expiry_date.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    print(f"  Keywords: {', '.join(fact.keywords)}")
                    print()
            else:
                print("\nNo facts found in database.")

        except Exception as e:
            logger.error(f"Error displaying facts: {str(e)}")
            print("Error retrieving facts.")

    async def start(self):
        print("\n🤖 Welcome to Enhanced AI Assistant v3.0")
        print("----------------------------------------")

        while True:
            try:
                choice = self._show_menu()

                if choice == "1":
                    name = input("\nEnter user name: ").strip()
                    if name:
                        self.current_user = self._create_user(name)
                        print(f"\n✅ Created user with ID: {self.current_user}")
                    else:
                        print("\n❌ Name cannot be empty")

                elif choice == "2":
                    user_id = input("\nEnter user ID: ").strip()
                    user = self.session.query(User).filter_by(
                        id=user_id).first()
                    if user:
                        self.current_user = user.id
                        print(f"\n✅ Selected user: {user.name}")
                    else:
                        print("\n❌ User not found")

                elif choice == "3":
                    if not self.current_user:
                        print("\n❌ Please select a user first")
                        continue

                    self.current_thread = self._create_thread(
                        self.current_user)
                    print(
                        f"\n📝 Starting new chat (Thread ID: {self.current_thread})"
                    )
                    print("Type 'exit' to return to menu")

                    while True:
                        try:
                            user_input = input("\n👤 You: ").strip()
                            if user_input.lower() == "exit":
                                break
                            if user_input:
                                response = await self.agent.process_message(
                                    user_input, self.current_thread,
                                    self.current_user)
                                print(f"\n🤖 Assistant: {response}")
                        except Exception as chat_error:
                            logger.error(f"Error in chat: {str(chat_error)}")
                            print(
                                "\n❌ Error processing message. Please try again."
                            )

                elif choice == "4":
                    if not self.current_user:
                        print("\n❌ Please select a user first")
                        continue
                    self._display_user_memories(self.current_user)

                elif choice == "5":
                    self._display_facts()

                elif choice == "6":
                    print("\n👋 Goodbye!")
                    break

                else:
                    print("\n❌ Invalid choice. Please select 1-6.")

            except Exception as e:
                logger.error(f"Error in menu operation: {str(e)}")
                print("\n❌ An error occurred. Please try again.")
                continue


# Main Function
async def main():
    required_vars = [
        "MISTRAL_API_KEY", "TAVILY_API_KEY", "PGDATABASE", "PGHOST", "PGPORT",
        "PGUSER", "PGPASSWORD"
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(
            f"❌ Missing required environment variables: {', '.join(missing)}")
        return 1

    try:
        # Initialize database tables
        Base.metadata.create_all(engine)
        logger.info("Database tables initialized")

        # Start chat interface
        chat = ChatInterface()
        await chat.start()
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\n❌ Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
