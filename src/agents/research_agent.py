"""
Research Agent implementation for the Research Assistant.

This module implements the main research agent that coordinates chains
and tools to perform intelligent research tasks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from src.chains.router_chain import ResearchRouterChain
from src.chains.sequential_chain import ResearchSequentialChain
from src.chains.transform_chain import ResearchTransformChain
from src.tools.browser_tool import BrowserTool
from src.tools.cite_tool import CiteTool, CitationManager
from src.tools.search_tool import SearchTool
from src.tools.summarize_tool import SummarizeTool
from src.utils.config import get_openrouter_llm, get_settings
from src.utils.helpers import format_timestamp, generate_id
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResearchSession:
    """Represents a research session with its context and results."""

    id: str = field(default_factory=generate_id)
    topic: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    queries: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    summary: str = ""
    report: str = ""

    def add_query(self, query: str) -> None:
        """Add a query to the session."""
        self.queries.append(query)

    def add_finding(self, finding: str) -> None:
        """Add a finding to the session."""
        self.findings.append(finding)

    def add_citation(self, citation: str) -> None:
        """Add a citation to the session."""
        self.citations.append(citation)

    def complete(self) -> None:
        """Mark the session as complete."""
        self.end_time = datetime.now()

    @property
    def duration(self) -> Optional[float]:
        """Get the session duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class ResearchReport(BaseModel):
    """Structured research report output."""

    title: str
    topic: str
    executive_summary: str
    key_findings: list[str]
    detailed_analysis: str
    citations: list[str]
    generated_at: str = Field(default_factory=format_timestamp)
    session_id: str = ""

    def to_markdown(self) -> str:
        """Convert the report to markdown format."""
        markdown = f"# {self.title}\n\n"
        markdown += f"**Topic:** {self.topic}\n\n"
        markdown += f"**Generated:** {self.generated_at}\n\n"
        markdown += "## Executive Summary\n\n"
        markdown += f"{self.executive_summary}\n\n"
        markdown += "## Key Findings\n\n"
        for i, finding in enumerate(self.key_findings, 1):
            markdown += f"{i}. {finding}\n"
        markdown += "\n## Detailed Analysis\n\n"
        markdown += f"{self.detailed_analysis}\n\n"
        markdown += "## References\n\n"
        for citation in self.citations:
            markdown += f"- {citation}\n"
        return markdown


class ResearchAgent:
    """
    An intelligent research agent that coordinates chains and tools.

    This agent orchestrates the research process by:
    1. Understanding the research topic
    2. Searching for relevant information
    3. Analyzing and summarizing findings
    4. Generating structured reports with citations

    Attributes:
        llm: The language model for agent reasoning.
        search_tool: Tool for web searches.
        summarize_tool: Tool for text summarization.
        cite_tool: Tool for citation management.
        browser_tool: Tool for web browsing.
        router_chain: Chain for routing queries to appropriate handlers.
        session: Current research session.
        reasoning_enabled: Whether OpenRouter reasoning feature is enabled.

    Example:
        >>> agent = ResearchAgent(llm=my_llm)
        >>> result = agent.research("Latest developments in quantum computing")
        >>> print(result.to_markdown())
    """

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        search_tool: Optional[SearchTool] = None,
        summarize_tool: Optional[SummarizeTool] = None,
        cite_tool: Optional[CiteTool] = None,
        browser_tool: Optional[BrowserTool] = None,
        use_openrouter: bool = True,
        model: Optional[str] = None,
        temperature: float = 0.7,
        reasoning_enabled: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Research Agent.

        Args:
            llm: Optional language model for agent reasoning. If not provided,
                a default model will be created using get_openrouter_llm() or
                get_llm() based on the use_openrouter parameter.
            search_tool: Optional custom search tool.
            summarize_tool: Optional custom summarize tool.
            cite_tool: Optional custom cite tool.
            browser_tool: Optional custom browser tool for web browsing.
            use_openrouter: If True (default), use OpenRouter API for the LLM.
                If False, use direct OpenAI API.
            model: Optional model name. If not provided, uses the configured
                default from environment variables.
            temperature: The temperature for the LLM (default: 0.7).
            reasoning_enabled: Whether to enable OpenRouter reasoning feature (default: False).
                When enabled, the model will provide reasoning traces for its responses.
                Only supported by specific models (e.g., minimax/minimax-m2.5, some Claude models).
            **kwargs: Additional keyword arguments passed to the LLM.
        """
        # Initialize LLM - either use provided one or create default
        if llm is None:
            if use_openrouter:
                self.llm = get_openrouter_llm(
                    model=model,
                    temperature=temperature,
                    reasoning_enabled=reasoning_enabled,
                    **kwargs,
                )
            else:
                from src.utils.config import get_llm
                self.llm = get_llm(
                    model=model,
                    temperature=temperature,
                    use_openrouter=False,
                    reasoning_enabled=reasoning_enabled,
                    **kwargs,
                )
        else:
            self.llm = llm

        self.search_tool = search_tool or SearchTool()
        self.summarize_tool = summarize_tool or SummarizeTool(llm=self.llm)
        self.cite_tool = cite_tool or CiteTool()
        self.browser_tool = browser_tool or BrowserTool()

        self._settings = get_settings()
        self._agent_executor: Optional[AgentExecutor] = None
        self._router_chain: Optional[ResearchRouterChain] = None
        self._sequential_chain: Optional[ResearchSequentialChain] = None
        self._transform_chain: Optional[ResearchTransformChain] = None

        self.session: Optional[ResearchSession] = None
        self._citation_manager = CitationManager()
        self.reasoning_enabled = reasoning_enabled

        logger.info(f"ResearchAgent initialized (reasoning_enabled={reasoning_enabled})")

    @classmethod
    def with_openrouter(
        cls,
        model: Optional[str] = None,
        temperature: float = 0.7,
        reasoning_enabled: bool = False,
        **kwargs: Any,
    ) -> "ResearchAgent":
        """
        Create a ResearchAgent configured for OpenRouter API.

        This is a convenience class method that creates an agent with
        OpenRouter configuration.

        Args:
            model: Optional model name. Uses OPENROUTER_MODEL from environment
                if not specified.
            temperature: The temperature for the LLM (default: 0.7).
            reasoning_enabled: Whether to enable OpenRouter reasoning feature (default: False).
                When enabled, the model will provide reasoning traces for its responses.
                Only supported by specific models (e.g., minimax/minimax-m2.5, some Claude models).
                Set to True or use OPENROUTER_REASONING_ENABLED=true environment variable.
            **kwargs: Additional keyword arguments passed to the LLM.

        Returns:
            A ResearchAgent instance configured for OpenRouter.

        Example:
            >>> agent = ResearchAgent.with_openrouter()
            >>> agent = ResearchAgent.with_openrouter(model="anthropic/claude-3-opus")
            >>> agent = ResearchAgent.with_openrouter(
            ...     model="minimax/minimax-m2.5",
            ...     reasoning_enabled=True
            ... )
        """
        return cls(
            use_openrouter=True,
            model=model,
            temperature=temperature,
            reasoning_enabled=reasoning_enabled,
            **kwargs,
        )

    @classmethod
    def with_openai(
        cls,
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> "ResearchAgent":
        """
        Create a ResearchAgent configured for direct OpenAI API.

        This is a convenience class method that creates an agent with
        direct OpenAI configuration.

        Args:
            model: Optional model name. Uses OPENAI_MODEL from environment
                if not specified.
            temperature: The temperature for the LLM (default: 0.7).
            **kwargs: Additional keyword arguments passed to the LLM.

        Returns:
            A ResearchAgent instance configured for direct OpenAI.

        Example:
            >>> agent = ResearchAgent.with_openai()
            >>> agent = ResearchAgent.with_openai(model="gpt-4-turbo-preview")
        """
        return cls(
            use_openrouter=False,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def _create_agent_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for the agent."""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an Intelligent Research Assistant. Your role is to help users "
                    "research topics by finding information, analyzing it, and generating "
                    "structured reports.\n\n"
                    "You have access to the following tools:\n"
                    "- search: Search the web for information\n"
                    "- browser: Browse and extract content from web pages (visit URLs, extract content, get links)\n"
                    "- summarize: Summarize text content\n"
                    "- cite: Create and manage citations\n\n"
                    "When researching:\n"
                    "1. First search for relevant information or browse specific URLs\n"
                    "2. Use the browser tool to visit and extract content from specific web pages\n"
                    "3. Analyze and summarize the findings\n"
                    "4. Create citations for all sources used\n"
                    "5. Generate a comprehensive report\n\n"
                    "Always provide accurate citations and be thorough in your research.",
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools."""
        tools = [self.search_tool, self.browser_tool, self.summarize_tool, self.cite_tool]

        prompt = self._create_agent_prompt()
        agent = create_openai_tools_agent(self.llm, tools, prompt)

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )

        logger.info("Agent executor created")
        return executor

    @property
    def agent_executor(self) -> AgentExecutor:
        """Get or create the agent executor."""
        if self._agent_executor is None:
            self._agent_executor = self._create_agent_executor()
        return self._agent_executor

    def start_session(self, topic: str) -> ResearchSession:
        """
        Start a new research session.

        Args:
            topic: The research topic.

        Returns:
            The new ResearchSession object.
        """
        self.session = ResearchSession(topic=topic)
        self.cite_tool.clear_citations()
        logger.info(f"Started research session for: {topic}")
        return self.session

    def research(self, query: str, generate_report: bool = True) -> ResearchReport:
        """
        Conduct research on a given query.

        Args:
            query: The research query or topic.
            generate_report: Whether to generate a full report.

        Returns:
            A ResearchReport with the findings.
        """
        # Start session if not already started
        if self.session is None:
            self.start_session(query)
        else:
            self.session.add_query(query)

        logger.info(f"Researching: {query}")

        try:
            # Execute the agent
            result = self.agent_executor.invoke({"input": query})
            output = result.get("output", "")

            # Process the output
            self.session.add_finding(output)

            # Generate report if requested
            if generate_report:
                return self.generate_report()
            else:
                return self._create_simple_report(output)

        except Exception as e:
            logger.error(f"Research failed: {str(e)}")
            raise

    def research_with_chains(
        self,
        query: str,
        use_router: bool = True,
        use_sequential: bool = False,
    ) -> dict[str, Any]:
        """
        Conduct research using chains for decision-making.

        Args:
            query: The research query.
            use_router: Whether to use the router chain.
            use_sequential: Whether to use the sequential chain.

        Returns:
            A dictionary with results and metadata.
        """
        if self.session is None:
            self.start_session(query)

        result: dict[str, Any] = {"query": query, "route": None}

        if use_router and self._router_chain:
            # Use router to determine best approach
            router_result = self._router_chain.invoke({self._router_chain.input_key: query})
            result["route"] = router_result.get("route", "unknown")
            result["content"] = router_result.get(self._router_chain.output_key, "")

        elif use_sequential and self._sequential_chain:
            # Use sequential chain for multi-step processing
            seq_result = self._sequential_chain.invoke(
                {self._sequential_chain.input_key: query}
            )
            result["content"] = seq_result.get(self._sequential_chain.output_key, "")

        else:
            # Fall back to agent executor
            agent_result = self.agent_executor.invoke({"input": query})
            result["content"] = agent_result.get("output", "")

        if self.session:
            self.session.add_finding(result.get("content", ""))

        return result

    def generate_report(self) -> ResearchReport:
        """
        Generate a comprehensive research report.

        Returns:
            A ResearchReport with all findings.
        """
        if not self.session:
            raise ValueError("No active research session")

        # Use LLM to generate structured report
        report_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Generate a structured research report based on the findings.",
                ),
                (
                    "human",
                    "Topic: {topic}\n\n"
                    "Findings:\n{findings}\n\n"
                    "Generate a report with:\n"
                    "1. A title\n"
                    "2. Executive summary\n"
                    "3. Key findings (as a list)\n"
                    "4. Detailed analysis",
                ),
            ]
        )

        findings_text = "\n\n".join(self.session.findings)
        prompt = report_prompt.format(
            topic=self.session.topic, findings=findings_text
        )

        response = self.llm.invoke(prompt)
        report_content = response.content if hasattr(response, "content") else str(response)

        # Get bibliography
        bibliography = self.cite_tool.get_bibliography("apa")

        report = ResearchReport(
            title=f"Research Report: {self.session.topic}",
            topic=self.session.topic,
            executive_summary=self._extract_section(report_content, "Executive Summary"),
            key_findings=self._extract_findings(report_content),
            detailed_analysis=self._extract_section(report_content, "Detailed Analysis")
            or report_content,
            citations=bibliography.split("\n") if bibliography else [],
            session_id=self.session.id,
        )

        if self.session:
            self.session.report = report.to_markdown()
            self.session.complete()

        logger.info("Research report generated")
        return report

    def _create_simple_report(self, content: str) -> ResearchReport:
        """Create a simple report from content."""
        topic = self.session.topic if self.session else "Research"

        return ResearchReport(
            title=f"Research: {topic}",
            topic=topic,
            executive_summary=content[:500] + "..." if len(content) > 500 else content,
            key_findings=[content[:200]],
            detailed_analysis=content,
            citations=self.cite_tool.get_bibliography("apa").split("\n"),
            session_id=self.session.id if self.session else generate_id(),
        )

    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a section from report content."""
        lines = content.split("\n")
        in_section = False
        section_lines = []

        for line in lines:
            if section_name.lower() in line.lower():
                in_section = True
                continue
            if in_section:
                if line.startswith("#") and section_name.lower() not in line.lower():
                    break
                section_lines.append(line)

        return "\n".join(section_lines).strip()

    def _extract_findings(self, content: str) -> list[str]:
        """Extract key findings from report content."""
        findings = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "-", "•", "*")):
                finding = line.lstrip("12345.-•* ").strip()
                if finding:
                    findings.append(finding)

        return findings[:5]  # Return top 5 findings

    def setup_chains(self) -> None:
        """Set up the chain infrastructure for the agent."""
        # Create router chain
        self._router_chain = ResearchRouterChain.create_research_router(
            llm=self.llm,
            search_chain=self._create_search_chain(),
            summarize_chain=self._create_summarize_chain(),
            analyze_chain=self._create_analyze_chain(),
        )

        # Create sequential chain
        self._sequential_chain = ResearchSequentialChain.create_research_chain(
            llm=self.llm
        )

        # Create transform chain
        self._transform_chain = ResearchTransformChain.create_format_chain(
            output_format="markdown", llm=self.llm
        )

        logger.info("Chain infrastructure set up")

    def _create_search_chain(self) -> Runnable:
        """Create a search chain."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Search for information about: {query}"),
                ("human", "{query}"),
            ]
        )
        return prompt | self.search_tool

    def _create_summarize_chain(self) -> Runnable:
        """Create a summarization chain."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Summarize the following: {query}"),
                ("human", "{query}"),
            ]
        )
        return prompt | self.summarize_tool

    def _create_analyze_chain(self) -> Runnable:
        """Create an analysis chain."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Analyze the following information and provide insights: {query}",
                ),
                ("human", "{query}"),
            ]
        )
        return prompt | self.llm

    def get_session_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current research session.

        Returns:
            A dictionary with session statistics.
        """
        if not self.session:
            return {"status": "no_active_session"}

        return {
            "session_id": self.session.id,
            "topic": self.session.topic,
            "queries_count": len(self.session.queries),
            "findings_count": len(self.session.findings),
            "citations_count": len(self.session.citations),
            "duration_seconds": self.session.duration,
            "status": "active" if self.session.end_time is None else "completed",
        }

    def browse_url(self, url: str) -> dict[str, Any]:
        """
        Browse a URL and extract its content using the browser tool.

        Args:
            url: The URL to browse.

        Returns:
            A dictionary with the browsing results including title, content, and links.

        Example:
            >>> agent = ResearchAgent()
            >>> result = agent.browse_url("https://example.com")
            >>> print(result["title"])
        """
        if self.session:
            self.session.add_query(f"Browsed: {url}")

        result = self.browser_tool.visit(url)

        if result.success:
            logger.info(f"Successfully browsed: {url}")
            if self.session:
                self.session.add_finding(f"Browsed {url}: {result.title}")
            return {
                "success": True,
                "url": result.url,
                "title": result.title,
                "content": result.content,
                "links": result.links,
                "metadata": result.metadata,
                "from_cache": result.from_cache,
            }
        else:
            logger.error(f"Failed to browse {url}: {result.error}")
            return {
                "success": False,
                "url": url,
                "error": result.error,
            }

    def reset(self) -> None:
        """Reset the agent state."""
        self.session = None
        self.cite_tool.clear_citations()
        self._citation_manager.clear()
        logger.info("Agent state reset")
