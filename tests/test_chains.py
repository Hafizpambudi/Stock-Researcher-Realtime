"""
Tests for chain modules.

This module contains unit tests for the sequential, transform, and router chains.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.runnables import RunnableLambda

from src.chains.router_chain import ResearchRouterChain, RouteResult
from src.chains.sequential_chain import ResearchSequentialChain
from src.chains.transform_chain import ResearchTransformChain, TransformConfig


class TestSequentialChain:
    """Test cases for ResearchSequentialChain."""

    def test_sequential_chain_initialization(self) -> None:
        """Test sequential chain initialization."""
        chain = ResearchSequentialChain()
        assert chain.name == "research_sequential_chain"
        assert chain.input_key == "query"
        assert chain.output_key == "result"

    def test_sequential_chain_input_output_keys(self) -> None:
        """Test input and output key properties."""
        chain = ResearchSequentialChain()
        assert chain.input_keys == ["query"]
        assert chain.output_keys == ["result"]

    def test_sequential_chain_empty_steps_error(self) -> None:
        """Test error when no steps are configured."""
        chain = ResearchSequentialChain()
        with pytest.raises(ValueError, match="No steps configured"):
            chain.invoke({"query": "test"})

    def test_sequential_chain_add_step(self) -> None:
        """Test adding steps to the chain."""
        chain = ResearchSequentialChain()

        step = RunnableLambda(lambda x: x.upper())
        chain.add_step(step)

        assert len(chain.steps) == 1

    def test_sequential_chain_remove_step(self) -> None:
        """Test removing steps from the chain."""
        chain = ResearchSequentialChain()

        step1 = RunnableLambda(lambda x: x.upper())
        step2 = RunnableLambda(lambda x: x + "!")

        chain.add_step(step1)
        chain.add_step(step2)

        assert len(chain.steps) == 2

        chain.remove_step(0)
        assert len(chain.steps) == 1

    def test_sequential_chain_execution(self) -> None:
        """Test sequential chain execution."""
        chain = ResearchSequentialChain()

        # Add simple transform steps
        chain.add_step(RunnableLambda(lambda x: {"result": x.get("query", "").upper()}))
        chain.add_step(RunnableLambda(lambda x: {"result": x.get("result", "") + "!"}))

        result = chain.invoke({"query": "hello"})
        assert result["result"] == "HELLO!"

    def test_create_research_chain(self) -> None:
        """Test creating pre-configured research chain."""
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=MagicMock(content="response"))

        chain = ResearchSequentialChain.create_research_chain(
            llm=mock_llm,
            include_search=True,
            include_analysis=True,
            include_summary=True,
        )

        assert len(chain.steps) == 3
        assert chain.llm == mock_llm


class TestTransformChain:
    """Test cases for ResearchTransformChain."""

    def test_transform_chain_initialization(self) -> None:
        """Test transform chain initialization."""
        chain = ResearchTransformChain()
        assert chain.name == "research_transform_chain"
        assert chain.input_key == "content"
        assert chain.output_key == "transformed"

    def test_transform_chain_input_output_keys(self) -> None:
        """Test input and output key properties."""
        chain = ResearchTransformChain()
        assert chain.input_keys == ["content"]
        assert chain.output_keys == ["transformed"]

    def test_transform_chain_execution(self) -> None:
        """Test transform chain execution."""
        chain = ResearchTransformChain(
            transforms=[
                lambda x: x.upper(),
                lambda x: x + "!",
            ]
        )

        result = chain.invoke({"content": "hello"})
        assert result["transformed"] == "HELLO!"

    def test_transform_chain_add_transform(self) -> None:
        """Test adding transforms."""
        chain = ResearchTransformChain()
        chain.add_transform(lambda x: x.lower())
        assert len(chain.transforms) == 1

    def test_transform_chain_clear_transforms(self) -> None:
        """Test clearing transforms."""
        chain = ResearchTransformChain(
            transforms=[lambda x: x.upper()]
        )
        assert len(chain.transforms) == 1

        chain.clear_transforms()
        assert len(chain.transforms) == 0

    def test_transform_config_functions(self) -> None:
        """Test built-in transform configurations."""
        assert TransformConfig.uppercase("hello") == "HELLO"
        assert TransformConfig.lowercase("HELLO") == "hello"
        assert TransformConfig.title_case("hello world") == "Hello World"
        assert TransformConfig.sanitize("hello   world") == "hello world"
        assert len(TransformConfig.truncate("hello", max_length=3)) <= 6

    def test_create_format_chain(self) -> None:
        """Test creating pre-configured format chain."""
        chain = ResearchTransformChain.create_format_chain(output_format="markdown")

        result = chain.invoke({"content": "test content"})
        assert "# Research Report" in result["transformed"]

    def test_create_format_chain_bullet_points(self) -> None:
        """Test creating bullet points format chain."""
        chain = ResearchTransformChain.create_format_chain(
            output_format="bullet_points"
        )

        text = "First point. Second point. Third point."
        result = chain.invoke({"content": text})
        assert "•" in result["transformed"] or "First" in result["transformed"]


class TestRouterChain:
    """Test cases for ResearchRouterChain."""

    def test_router_chain_initialization(self) -> None:
        """Test router chain initialization."""
        chain = ResearchRouterChain()
        assert chain.name == "research_router_chain"
        assert chain.input_key == "query"
        assert chain.output_key == "result"

    def test_router_chain_empty_routes_error(self) -> None:
        """Test error when no routes are configured."""
        chain = ResearchRouterChain()
        with pytest.raises(ValueError, match="No routes configured"):
            chain.invoke({"query": "test"})

    def test_router_chain_add_route(self) -> None:
        """Test adding routes."""
        chain = ResearchRouterChain()

        mock_chain = RunnableLambda(lambda x: {"result": "response"})
        chain.add_route("search", mock_chain, "Search for information")

        assert "search" in chain.routes
        assert "search" in chain.route_descriptions

    def test_router_chain_remove_route(self) -> None:
        """Test removing routes."""
        chain = ResearchRouterChain()

        mock_chain = RunnableLambda(lambda x: {"result": "response"})
        chain.add_route("search", mock_chain)
        chain.add_route("summarize", mock_chain)

        assert len(chain.routes) == 2

        chain.remove_route("search")
        assert len(chain.routes) == 1
        assert "search" not in chain.routes

    def test_router_chain_get_available_routes(self) -> None:
        """Test getting available routes."""
        chain = ResearchRouterChain()

        mock_chain = RunnableLambda(lambda x: {"result": "response"})
        chain.add_route("search", mock_chain)
        chain.add_route("summarize", mock_chain)

        routes = chain.get_available_routes()
        assert "search" in routes
        assert "summarize" in routes

    def test_router_chain_keyword_routing(self) -> None:
        """Test keyword-based routing."""
        chain = ResearchRouterChain()

        mock_chain = RunnableLambda(lambda x: {"result": "response"})
        chain.add_route("search", mock_chain)
        chain.add_route("summarize", mock_chain)
        chain.add_route("analyze", mock_chain)

        # Test search keywords
        result = chain._keyword_route("Find information about AI")
        assert result.route_name == "search"

        # Test summarize keywords
        result = chain._keyword_route("Summarize this article")
        assert result.route_name == "summarize"

    def test_router_chain_default_route(self) -> None:
        """Test default route fallback."""
        chain = ResearchRouterChain(default_route="search")

        mock_chain = RunnableLambda(lambda x: {"result": "response"})
        chain.add_route("search", mock_chain)

        result = chain._keyword_route("random query xyz")
        assert result.route_name == "search"

    def test_route_result(self) -> None:
        """Test RouteResult class."""
        mock_chain = RunnableLambda(lambda x: {"result": "response"})

        result = RouteResult(
            route_name="test",
            chain=mock_chain,
            confidence=0.9,
            reasoning="Test reasoning",
        )

        assert result.route_name == "test"
        assert result.confidence == 0.9
        assert result.reasoning == "Test reasoning"

    def test_create_research_router(self) -> None:
        """Test creating pre-configured research router."""
        mock_llm = MagicMock()

        search_chain = RunnableLambda(lambda x: {"result": "search result"})
        summarize_chain = RunnableLambda(lambda x: {"result": "summary"})

        router = ResearchRouterChain.create_research_router(
            llm=mock_llm,
            search_chain=search_chain,
            summarize_chain=summarize_chain,
        )

        assert "search" in router.routes
        assert "summarize" in router.routes
        assert router.router_llm == mock_llm
        assert router.default_route == "search"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
