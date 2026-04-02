"""
Pytest configuration and fixtures for Research Assistant tests.

This module provides shared fixtures and configuration for all tests.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=MagicMock(content="Mock response"))
    llm.ainvoke = MagicMock(return_value=MagicMock(content="Mock async response"))
    return llm


@pytest.fixture
def mock_search_results() -> dict:
    """Mock search results for testing."""
    return {
        "results": [
            {
                "title": "Test Result 1",
                "link": "https://example.com/1",
                "snippet": "This is a test search result.",
            },
            {
                "title": "Test Result 2",
                "link": "https://example.com/2",
                "snippet": "Another test search result.",
            },
        ]
    }


@pytest.fixture
def sample_article_text() -> str:
    """Sample article text for testing."""
    return """
    Artificial intelligence is transforming industries worldwide.
    Machine learning algorithms can now process vast amounts of data.
    Deep learning has enabled breakthroughs in image recognition.
    Natural language processing allows computers to understand human language.
    The future of AI holds immense promise and potential challenges.
    """ * 10


@pytest.fixture
def sample_citation_data() -> dict:
    """Sample citation data for testing."""
    return {
        "content": "https://example.com/article",
        "citation_type": "web",
        "metadata": {
            "title": "Test Article",
            "author": "John Doe",
            "date": "2024-01-15",
        },
    }


@pytest.fixture
def clean_env() -> None:
    """Fixture to clean environment variables before test."""
    # Store original environment
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def setup_test_environment(clean_env: None) -> None:
    """Set up test environment variables."""
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["LOG_LEVEL"] = "DEBUG"


# Configure pytest-asyncio if available
try:
    import pytest_asyncio

    pytest_plugins = ["pytest_asyncio"]
except ImportError:
    pass
