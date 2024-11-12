from langchain_core.tools import Tool
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime

class ResearchResult(BaseModel):
    """Structure for research results"""
    topic: str
    key_points: List[str]
    sources: List[str]
    relevance_score: float


class SectionContent(BaseModel):
    """Structure for section content"""
    title: str
    content: str
    citations: List[str]
    metadata: Dict


def create_article_tools(article_progress):
    """Creates the tools used by the article writing agent"""

    def save_research(research_data: str) -> str:
        """Save research findings"""
        article_progress.research_data.append({
            "data":
            research_data,
            "timestamp":
            datetime.now().isoformat()
        })
        return f"Research data saved successfully"

    def save_section(content: str, section_title: str) -> str:
        """Save a section of the article"""
        version = ArticleVersion(content=content,
                                 step=article_progress.current_step,
                                 section=section_title)
        article_progress.add_version(version)
        return f"Saved section: {section_title}"

    def analyze_feedback(feedback: str) -> str:
        """Analyze and incorporate feedback"""
        article_progress.add_feedback(feedback)
        return "Feedback analyzed and recorded"

    tools = [
        Tool(name="save_research",
             description="Save research findings for the article",
             func=save_research),
        Tool(name="save_section",
             description="Save a section of the article",
             func=save_section),
        Tool(name="analyze_feedback",
             description="Analyze and incorporate feedback",
             func=analyze_feedback)
    ]

    return tools
