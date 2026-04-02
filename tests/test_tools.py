"""
Tests for tool modules.

This module contains unit tests for the search, summarize, and cite tools.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.cite_tool import CiteTool, Citation, CitationManager
from src.tools.search_tool import SearchTool, SearchResult
from src.tools.summarize_tool import SummarizeTool


class TestSearchTool:
    """Test cases for SearchTool."""

    def test_search_tool_initialization(self) -> None:
        """Test SearchTool initialization."""
        tool = SearchTool()
        assert tool.name == "search"
        assert "search" in tool.description.lower()

    def test_search_tool_input_schema(self) -> None:
        """Test SearchTool input validation."""
        tool = SearchTool()
        # Should have query field
        assert tool.args_schema is not None

    def test_search_mock_results(self) -> None:
        """Test mock search results when no engine is configured."""
        tool = SearchTool()
        tool._search_engine = None  # Force mock mode

        results = tool._mock_search("test query", 3)
        assert len(results) == 3
        assert all("test query" in r["title"] for r in results)

    def test_search_tool_run(self) -> None:
        """Test SearchTool run method."""
        tool = SearchTool()
        tool._search_engine = None  # Force mock mode

        result = tool.run({"query": "test query"})
        assert isinstance(result, str)
        assert "Search Result" in result or "No results" in result

    def test_search_with_metadata(self) -> None:
        """Test search with metadata output."""
        tool = SearchTool()
        tool._search_engine = None  # Force mock mode

        results = tool.search_with_metadata("test query", num_results=2)
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], SearchResult)


class TestSummarizeTool:
    """Test cases for SummarizeTool."""

    def test_summarize_tool_initialization(self) -> None:
        """Test SummarizeTool initialization."""
        tool = SummarizeTool()
        assert tool.name == "summarize"
        assert "summarize" in tool.description.lower()

    def test_summarize_tool_input_schema(self) -> None:
        """Test SummarizeTool input validation."""
        tool = SummarizeTool()
        assert tool.args_schema is not None

    def test_summarize_without_llm(self) -> None:
        """Test summarization fallback without LLM."""
        tool = SummarizeTool()
        tool.llm = None

        text = "This is a test. It has multiple sentences. " * 10
        result = tool.run({"text": text, "style": "concise"})

        assert isinstance(result, str)
        # Should return extractive summary
        assert len(result) > 0

    def test_summarize_short_text(self) -> None:
        """Test summarization of short text."""
        tool = SummarizeTool()
        tool.llm = None

        text = "Short text."
        result = tool.run({"text": text})
        assert isinstance(result, str)

    def test_summarize_set_llm(self) -> None:
        """Test setting LLM on summarize tool."""
        tool = SummarizeTool()
        mock_llm = MagicMock()
        tool.set_llm(mock_llm)
        assert tool.llm == mock_llm


class TestCiteTool:
    """Test cases for CiteTool."""

    def test_cite_tool_initialization(self) -> None:
        """Test CiteTool initialization."""
        tool = CiteTool()
        assert tool.name == "cite"
        assert "citation" in tool.description.lower()

    def test_cite_tool_input_schema(self) -> None:
        """Test CiteTool input validation."""
        tool = CiteTool()
        assert tool.args_schema is not None

    def test_create_citation(self) -> None:
        """Test citation creation."""
        tool = CiteTool()

        citation = tool._create_citation(
            content="https://example.com/article",
            citation_type="web",
            metadata={"title": "Test Article", "author": "John Doe"},
        )

        assert citation.citation_type == "web"
        assert citation.url == "https://example.com/article"
        assert citation.title == "Test Article"
        assert citation.author == "John Doe"
        assert citation.citation_key.startswith("web_")

    def test_cite_tool_run(self) -> None:
        """Test CiteTool run method."""
        tool = CiteTool()

        result = tool.run({
            "content": "https://example.com",
            "citation_type": "web",
        })

        assert isinstance(result, str)
        assert "[" in result  # Should contain citation key

    def test_citation_manager(self) -> None:
        """Test CitationManager functionality."""
        manager = CitationManager()

        citation = Citation(
            id="test-id",
            content="Test content",
            citation_type="web",
            url="https://example.com",
            title="Test",
            author="Author",
            date="2024-01-01",
            accessed_date="2024-01-15",
            citation_key="web_test",
        )

        key = manager.add_citation(citation)
        assert key == "web_test"

        retrieved = manager.get_citation("test-id")
        assert retrieved == citation

        all_citations = manager.get_all_citations()
        assert len(all_citations) == 1

    def test_citation_formatting(self) -> None:
        """Test citation format methods."""
        citation = Citation(
            id="test-id",
            content="Test content",
            citation_type="web",
            url="https://example.com",
            title="Test Article",
            author="John Doe",
            date="2024-01-01",
            accessed_date="2024-01-15",
            citation_key="web_test",
        )

        apa = citation.format_apa()
        assert "John Doe" in apa
        assert "2024" in apa

        mla = citation.format_mla()
        assert "John Doe" in mla
        assert "Test Article" in mla

    def test_bibliography_generation(self) -> None:
        """Test bibliography generation."""
        tool = CiteTool()

        # Add multiple citations
        tool.run({"content": "https://example1.com", "citation_type": "web"})
        tool.run({"content": "https://example2.com", "citation_type": "web"})

        bibliography = tool.get_bibliography("apa")
        assert isinstance(bibliography, str)
        assert "[1]" in bibliography or "1" in bibliography

    def test_citation_count(self) -> None:
        """Test citation count tracking."""
        tool = CiteTool()
        assert tool.get_citation_count() == 0

        tool.run({"content": "https://example.com", "citation_type": "web"})
        assert tool.get_citation_count() == 1

    def test_clear_citations(self) -> None:
        """Test clearing citations."""
        tool = CiteTool()
        tool.run({"content": "https://example.com", "citation_type": "web"})
        assert tool.get_citation_count() == 1

        tool.clear_citations()
        assert tool.get_citation_count() == 0

    def test_cite_multiple(self) -> None:
        """Test creating multiple citations at once."""
        tool = CiteTool()

        sources = [
            {"content": "https://example1.com", "citation_type": "web"},
            {"content": "https://example2.com", "citation_type": "web"},
        ]

        results = tool.cite_multiple(sources)
        assert len(results) == 2
        assert tool.get_citation_count() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
