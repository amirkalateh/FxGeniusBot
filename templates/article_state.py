from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from datetime import datetime
import json


class ArticleState(TypedDict):
    """Enhanced state management for article writing process"""
    messages: Annotated[List[BaseMessage], add_messages]
    current_version: str
    current_step: str
    current_section: dict
    feedback_history: List[dict]
    research_data: List[dict]
    outline: Optional[dict]
    is_last_step: bool
    metadata: dict  # For tracking progress, versions, etc.


class ArticleVersion:
    """Represents a version of the article content"""

    def __init__(self, content: str, step: str, section: str = None):
        self.content = content
        self.step = step
        self.section = section
        self.timestamp = datetime.now().isoformat()
        self.metadata = {}

    def to_dict(self):
        return {
            "content": self.content,
            "step": self.step,
            "section": self.section,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class ArticleProgress:
    """Tracks the progress of article writing"""

    def __init__(self):
        self.versions = []
        self.current_step = "start"
        self.sections_completed = set()
        self.feedback_history = []
        self.research_data = []

    def add_version(self, version: ArticleVersion):
        self.versions.append(version.to_dict())

    def add_feedback(self, feedback: str, section: str = None):
        self.feedback_history.append({
            "feedback": feedback,
            "section": section,
            "timestamp": datetime.now().isoformat()
        })

    def get_latest_version(self) -> Optional[dict]:
        return self.versions[-1] if self.versions else None
