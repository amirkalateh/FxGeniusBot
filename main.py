from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent, InjectedStore
from typing import Annotated, Any
from langchain_core.runnables import RunnableConfig
import json
import os

# Message history storage
class MessageHistory:
    def __init__(self, file_path: str = "message_history.json"):
        self.file_path = file_path
        self.messages = self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                messages = []
                # Always start with system message
                system_messages = [m for m in data if m["type"] == "system"]
                if system_messages:
                    messages.append(SystemMessage(content=system_messages[0]["content"]))
                # Then add other messages
                for m in data:
                    if m["type"] != "system":
                        if m["type"] == "human":
                            messages.append(HumanMessage(content=m["content"]))
                        elif m["type"] == "ai":
                            messages.append(AIMessage(content=m["content"]))
                return messages
        return [SystemMessage(content="You are a helpful AI assistant. Remember important information about users.")]

    def save(self):
        data = [
            {
                "type": "human" if isinstance(m, HumanMessage)
                else "agent" if isinstance(m, AIMessage)
                else "system",
                "content": m.content
            }
            for m in self.messages
        ]
        with open(self.file_path, 'w') as f:
            json.dump(data, f)

    def add_message(self, message):
        if isinstance(message, SystemMessage):
            # Replace existing system message if any
            if any(isinstance(m, SystemMessage) for m in self.messages):
                self.messages = [m for m in self.messages if not isinstance(m, SystemMessage)]
            # Add system message at the beginning
            self.messages.insert(0, message)
        else:
            self.messages.append(message)
        self.save()

    def get_messages(self):
        # Ensure system message is first
        if not any(isinstance(m, SystemMessage) for m in self.messages):
            self.messages.insert(0, SystemMessage(content="You are a helpful AI assistant. Remember important information about users."))
        return self.messages

# Create history instance
history = MessageHistory()

# Create memory store for cross-thread persistence
store = InMemoryStore()

@tool
def save_memory(memory: str, *, config: RunnableConfig, store: Annotated[Any, InjectedStore()]) -> str:
    """Save important information about the user"""
    # Don't add system messages to history for memory saves
    return f"Saved memory: {memory}"

def prepare_model_inputs(state: dict):
    """Convert the state into a format the model expects"""
    # Get historical messages (system message will be first)
    all_messages = history.get_messages()

    # Add current messages (excluding system messages)
    current_messages = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    all_messages.extend(current_messages)

    return all_messages

# Create the agent
memory = MemorySaver()
model = ChatAnthropic(model_name="claude-3-haiku-20240307")
search = TavilySearchResults(max_results=3)
tools = [search, save_memory]

agent_executor = create_react_agent(
    model, 
    tools,
    checkpointer=memory,
    store=store,
    state_modifier=prepare_model_inputs
)

# Use the agent
config = {
    "configurable": {
        "thread_id": "anc123",
        "user_id": "user2"
    }
}

def process_response(chunk):
    if isinstance(chunk, dict) and "messages" in chunk:
        for message in chunk["messages"]:
            print(message)
            if not isinstance(message, SystemMessage):
                history.add_message(message)
    print("----")

# Run conversations
while True:
    user_input = input("\nEnter your message (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break

    print("\nProcessing your message:")
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=user_input)]}, 
        config
    ):
        process_response(chunk)