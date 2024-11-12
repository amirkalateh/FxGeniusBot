from typing import List, Dict, Optional
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from article_state import ArticleState, ArticleProgress, ArticleVersion
from article_tools import create_article_tools
import json
import os

class InteractiveArticleWriter:
    """Main class for interactive article writing system"""

    def __init__(self):
        self.progress = ArticleProgress()
        self.llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0.7
        )
        self.search = TavilySearchResults(max_results=3)
        self.setup_agent()
        self.current_thread_id = None

    def setup_agent(self):
        """Initialize the agent with tools and graph structure"""
        # Combine search and article tools
        tools = [self.search] + create_article_tools(self.progress)

        # Create graph builder
        builder = StateGraph(ArticleState)

        # Create agent with enhanced prompt
        agent = create_react_agent(
            self.llm,
            tools,
            state_modifier=self._get_system_prompt()
        )

        # Build graph structure
        builder.add_node("agent", agent)
        builder.add_node("tools", ToolNode(tools))
        builder.set_entry_point("agent")

        builder.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: END,
            }
        )

        builder.add_edge("tools", "agent")

        # Compile graph
        self.graph = builder.compile(
            checkpointer=MemorySaver()
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        return """You are an expert research assistant and article writer.

        Current Step: {current_step}
        Section: {current_section}

        Guidelines:
        1. Focus only on the current step/section
        2. Use appropriate tools to save progress
        3. Be thorough but concise
        4. Wait for user approval before proceeding
        5. Incorporate feedback thoughtfully

        Always explain your thinking and next steps.
        """

    def process_step(self, instruction: str, step_name: str, section: Optional[str] = None):
        """Process a single step with user interaction"""
        self.progress.current_step = step_name

        config = {
            "configurable": {
                "thread_id": self.current_thread_id
            },
            "recursion_limit": 100
        }

        initial_state = {
            "messages": [("user", instruction)],
            "current_version": "",
            "current_step": step_name,
            "current_section": {"name": section} if section else {},
            "feedback_history": self.progress.feedback_history,
            "research_data": self.progress.research_data,
            "outline": None,
            "is_last_step": False,
            "metadata": {}
        }

        return self.graph.stream(
            initial_state,
            config,
            stream_mode="values"
        )

    def write_article(self, topic: str):
        """Main method to write an article interactively"""
        self.current_thread_id = f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Step 1: Initial Research
            self._handle_research_phase(topic)

            # Step 2: Create and Refine Outline
            outline = self._handle_outline_phase()

            # Step 3: Section Writing
            self._handle_writing_phase(outline)

            # Step 4: Final Review
            self._handle_final_review()

        except Exception as e:
            print(f"\nError during article writing: {str(e)}")
            self._save_progress()  # Save progress before exiting
            raise

    def _handle_research_phase(self, topic: str):
        """Handle the research phase"""
        print("\n=== Phase 1: Research ===")

        instruction = (
            f"Research the topic: {topic}\n"
            "1. Use Tavily search to find recent information\n"
            "2. Identify key points and trends\n"
            "3. Save research findings\n"
            "4. Summarize findings clearly"
        )

        for event in self.process_step(instruction, "research"):
            self._handle_event_output(event)

        if not self._get_user_approval("Research phase"):
            raise ValueError("Research phase not approved")

    def _handle_outline_phase(self) -> dict:
        """Handle the outline creation and refinement"""
        print("\n=== Phase 2: Outline Creation ===")

        while True:
            instruction = (
                "Create a detailed outline based on the research.\n"
                "Include:\n"
                "1. Main sections\n"
                "2. Subsections\n"
                "3. Key points for each section"
            )

            for event in self.process_step(instruction, "outline"):
                self._handle_event_output(event)

            feedback = self._get_user_feedback("outline")
            if feedback.lower() == 'approve':
                break

            print("\nRevising outline based on feedback...")

        return self.progress.get_latest_version()

    def _handle_writing_phase(self, outline: dict):
        """Handle the section-by-section writing"""
        print("\n=== Phase 3: Writing Process ===")

        sections = self._parse_outline_sections(outline['content'])

        for section_num, section in enumerate(sections, 1):
            print(f"\n--- Writing Section {section_num}: {section['title']} ---")

            if not self._get_user_approval(f"Ready to write section {section_num}"):
                break

            self._write_section(section, section_num)

            while True:
                feedback = self._get_user_feedback(f"section_{section_num}")
                if feedback.lower() == 'approve':
                    break
                self._revise_section(section, section_num, feedback)

    def _write_section(self, section: dict, section_num: int):
        """Write a single section"""
        instruction = (
            f"Write section {section_num}: {section['title']}\n"
            "Include:\n"
            f"- Main points: {', '.join(section['points'])}\n"
            "- Proper citations\n"
            "- Clear transitions"
        )

        for event in self.process_step(
            instruction,
            f"write_section_{section_num}",
            section['title']
        ):
            self._handle_event_output(event)

    def _revise_section(self, section: dict, section_num: int, feedback: str):
        """Revise a section based on feedback"""
        instruction = (
            f"Revise section {section_num} based on feedback:\n{feedback}\n"
            "Maintain:\n"
            "- Section coherence\n"
            "- Citation accuracy\n"
            "- Writing quality"
        )

        for event in self.process_step(
            instruction,
            f"revise_section_{section_num}",
            section['title']
        ):
            self._handle_event_output(event)

    def _handle_final_review(self):
        """Handle the final review phase"""
        print("\n=== Final Review ===")

        print("\nArticle Versions:")
        for version in self.progress.versions:
            print(f"\nStep: {version['step']}")
            print(f"Timestamp: {version['timestamp']}")
            if version['section']:
                print(f"Section: {version['section']}")
            print(f"Content Preview: {version['content'][:200]}...")

    def _handle_event_output(self, event: dict):
        """Handle output from events"""
        if "messages" in event:
            message = event["messages"][-1]
            if hasattr(message, 'content'):
                print("\nOutput:")
                print("=" * 50)
                print(message.content)
                print("=" * 50)

    def _get_user_approval(self, phase: str) -> bool:
        """Get user approval for a phase"""
        response = input(f"\nApprove {phase}? (yes/no): ").strip().lower()
        return response == 'yes'

    def _get_user_feedback(self, section: str) -> str:
        """Get user feedback for a section"""
        return input(f"\nProvide feedback for {section} (or type 'approve'): ").strip()

    def _parse_outline_sections(self, outline: str) -> List[Dict]:
        """Parse outline into structured sections"""
        # Implementation depends on outline format
        # This is a simple example
        sections = []
        current_section = None

        for line in outline.split('\n'):
            if line.strip().startswith('I') or line.strip().startswith('V'):
                if current_section:
                    sections.append(current_section)
                current_section = {'title': line.strip(), 'points': []}
            elif line.strip().startswith(('A', 'B', 'C')) and current_section:
                current_section['points'].append(line.strip())

        if current_section:
            sections.append(current_section)

        return sections

    def _save_progress(self):
        """Save current progress to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"article_progress_{timestamp}.json"

        data = {
            "versions": self.progress.versions,
            "feedback_history": self.progress.feedback_history,
            "research_data": self.progress.research_data,
            "metadata": {
                "last_step": self.progress.current_step,
                "sections_completed": list(self.progress.sections_completed)
            }
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nProgress saved to: {filename}")

def main():
    """Main execution function"""
    try:
        writer = InteractiveArticleWriter()

        print("Welcome to Interactive Article Writer!")
        topic = input("\nEnter article topic: ").strip()

        writer.write_article(topic)

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nPlease check your API keys:")
        print("- ANTHROPIC_API_KEY")
        print("- TAVILY_API_KEY")

if __name__ == "__main__":
    main()