"""
Tests for agent modules.

This module contains unit tests for the research agent.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.research_agent import ResearchAgent, ResearchReport, ResearchSession


class TestResearchSession:
    """Test cases for ResearchSession."""

    def test_session_initialization(self) -> None:
        """Test session initialization."""
        session = ResearchSession(topic="Test Topic")

        assert session.topic == "Test Topic"
        assert session.id is not None
        assert len(session.queries) == 0
        assert session.end_time is None

    def test_session_add_query(self) -> None:
        """Test adding queries to session."""
        session = ResearchSession()

        session.add_query("Query 1")
        session.add_query("Query 2")

        assert len(session.queries) == 2
        assert session.queries[0] == "Query 1"

    def test_session_add_finding(self) -> None:
        """Test adding findings to session."""
        session = ResearchSession()

        session.add_finding("Finding 1")
        session.add_finding("Finding 2")

        assert len(session.findings) == 2

    def test_session_add_citation(self) -> None:
        """Test adding citations to session."""
        session = ResearchSession()

        session.add_citation("Citation 1")
        session.add_citation("Citation 2")

        assert len(session.citations) == 2

    def test_session_complete(self) -> None:
        """Test completing a session."""
        session = ResearchSession()
        assert session.end_time is None

        session.complete()
        assert session.end_time is not None

    def test_session_duration(self) -> None:
        """Test session duration calculation."""
        session = ResearchSession()
        assert session.duration is None

        session.complete()
        assert session.duration is not None
        assert session.duration >= 0


class TestResearchReport:
    """Test cases for ResearchReport."""

    def test_report_initialization(self) -> None:
        """Test report initialization."""
        report = ResearchReport(
            title="Test Report",
            topic="Test Topic",
            executive_summary="Summary",
            key_findings=["Finding 1"],
            detailed_analysis="Analysis",
            citations=["Citation 1"],
        )

        assert report.title == "Test Report"
        assert report.topic == "Test Topic"
        assert len(report.key_findings) == 1
        assert report.generated_at is not None

    def test_report_to_markdown(self) -> None:
        """Test report markdown conversion."""
        report = ResearchReport(
            title="Test Report",
            topic="Test Topic",
            executive_summary="Summary",
            key_findings=["Finding 1", "Finding 2"],
            detailed_analysis="Detailed analysis content",
            citations=["Citation 1"],
        )

        markdown = report.to_markdown()

        assert "# Test Report" in markdown
        assert "**Topic:** Test Topic" in markdown
        assert "## Executive Summary" in markdown
        assert "Summary" in markdown
        assert "## Key Findings" in markdown
        assert "Finding 1" in markdown
        assert "## Detailed Analysis" in markdown
        assert "## References" in markdown


class TestResearchAgent:
    """Test cases for ResearchAgent."""

    def test_agent_initialization(self) -> None:
        """Test agent initialization."""
        mock_llm = MagicMock()

        agent = ResearchAgent(llm=mock_llm)

        assert agent.llm == mock_llm
        assert agent.search_tool is not None
        assert agent.summarize_tool is not None
        assert agent.cite_tool is not None
        assert agent.session is None

    def test_agent_start_session(self) -> None:
        """Test starting a research session."""
        mock_llm = MagicMock()
        agent = ResearchAgent(llm=mock_llm)

        session = agent.start_session("Test Topic")

        assert session.topic == "Test Topic"
        assert agent.session == session

    def test_agent_reset(self) -> None:
        """Test resetting the agent."""
        mock_llm = MagicMock()
        agent = ResearchAgent(llm=mock_llm)

        agent.start_session("Test Topic")
        agent.cite_tool.run({"content": "https://example.com", "citation_type": "web"})

        assert agent.session is not None
        assert agent.cite_tool.get_citation_count() > 0

        agent.reset()

        assert agent.session is None
        assert agent.cite_tool.get_citation_count() == 0

    def test_agent_get_session_summary_no_session(self) -> None:
        """Test session summary when no session exists."""
        mock_llm = MagicMock()
        agent = ResearchAgent(llm=mock_llm)

        summary = agent.get_session_summary()
        assert summary["status"] == "no_active_session"

    def test_agent_get_session_summary_with_session(self) -> None:
        """Test session summary with active session."""
        mock_llm = MagicMock()
        agent = ResearchAgent(llm=mock_llm)

        agent.start_session("Test Topic")
        agent.session.add_query("Query 1")
        agent.session.add_finding("Finding 1")

        summary = agent.get_session_summary()

        assert summary["status"] == "active"
        assert summary["topic"] == "Test Topic"
        assert summary["queries_count"] == 1
        assert summary["findings_count"] == 1

    def test_agent_create_simple_report(self) -> None:
        """Test creating a simple report."""
        mock_llm = MagicMock()
        agent = ResearchAgent(llm=mock_llm)

        agent.start_session("Test Topic")
        report = agent._create_simple_report("Test content")

        assert isinstance(report, ResearchReport)
        assert "Test Topic" in report.title
        assert report.detailed_analysis == "Test content"

    def test_agent_extract_section(self) -> None:
        """Test extracting sections from report content."""
        mock_llm = MagicMock()
        agent = ResearchAgent(llm=mock_llm)

        content = """
# Executive Summary
This is the summary.

# Detailed Analysis
This is the analysis.

# Other Section
Other content.
"""

        summary = agent._extract_section(content, "Executive Summary")
        assert "This is the summary" in summary

        analysis = agent._extract_section(content, "Detailed Analysis")
        assert "This is the analysis" in analysis

    def test_agent_extract_findings(self) -> None:
        """Test extracting findings from report content."""
        mock_llm = MagicMock()
        agent = ResearchAgent(llm=mock_llm)

        content = """
1. First finding
2. Second finding
3. Third finding
- Bullet finding
* Another bullet
"""

        findings = agent._extract_findings(content)
        assert len(findings) >= 3
        assert "First finding" in findings

    def test_agent_setup_chains(self) -> None:
        """Test setting up chain infrastructure."""
        mock_llm = MagicMock()
        agent = ResearchAgent(llm=mock_llm)

        # Should not raise
        agent.setup_chains()

        assert agent._router_chain is not None
        assert agent._sequential_chain is not None
        assert agent._transform_chain is not None

    @patch("src.agents.research_agent.create_openai_tools_agent")
    def test_agent_executor_creation(self, mock_create_agent: MagicMock) -> None:
        """Test agent executor creation."""
        mock_llm = MagicMock()
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        agent = ResearchAgent(llm=mock_llm)

        # Access agent_executor to trigger creation
        executor = agent.agent_executor

        assert executor is not None
        mock_create_agent.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
