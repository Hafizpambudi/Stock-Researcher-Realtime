"""
Tests for the BrowserTool.

This module contains comprehensive tests for the browser tool,
including URL visiting, content extraction, error handling, and rate limiting.
"""

import time
from unittest.mock import Mock, patch

import pytest
import requests

from src.tools.browser_tool import (
    BrowserResult,
    BrowserTool,
    LRUCache,
    RateLimiter,
)


class TestLRUCache:
    """Tests for the LRUCache class."""

    def test_cache_put_and_get(self) -> None:
        """Test basic cache put and get operations."""
        cache = LRUCache(max_size=5)

        cache.put(
            url="https://example.com",
            content="Test content",
            title="Test Title",
            links=["https://example.com/link1"],
            metadata={"key": "value"},
        )

        cached = cache.get("https://example.com")

        assert cached is not None
        assert cached.content == "Test content"
        assert cached.title == "Test Title"
        assert cached.links == ["https://example.com/link1"]
        assert cached.metadata == {"key": "value"}

    def test_cache_miss(self) -> None:
        """Test cache miss returns None."""
        cache = LRUCache(max_size=5)

        cached = cache.get("https://nonexistent.com")

        assert cached is None

    def test_cache_max_size(self) -> None:
        """Test that cache respects max size limit."""
        cache = LRUCache(max_size=3)

        cache.put("https://example1.com", "content1", "Title1", [], {})
        cache.put("https://example2.com", "content2", "Title2", [], {})
        cache.put("https://example3.com", "content3", "Title3", [], {})
        cache.put("https://example4.com", "content4", "Title4", [], {})

        # First entry should be evicted
        assert cache.get("https://example1.com") is None
        assert cache.get("https://example2.com") is not None
        assert cache.get("https://example3.com") is not None
        assert cache.get("https://example4.com") is not None
        assert len(cache) == 3

    def test_cache_lru_eviction(self) -> None:
        """Test that least recently used entry is evicted."""
        cache = LRUCache(max_size=3)

        cache.put("https://example1.com", "content1", "Title1", [], {})
        cache.put("https://example2.com", "content2", "Title2", [], {})
        cache.put("https://example3.com", "content3", "Title3", [], {})

        # Access example1 to make it recently used
        cache.get("https://example1.com")

        # Add new entry - example2 should be evicted (least recently used)
        cache.put("https://example4.com", "content4", "Title4", [], {})

        assert cache.get("https://example1.com") is not None
        assert cache.get("https://example2.com") is None
        assert cache.get("https://example3.com") is not None
        assert cache.get("https://example4.com") is not None

    def test_cache_clear(self) -> None:
        """Test cache clear operation."""
        cache = LRUCache(max_size=5)

        cache.put("https://example.com", "content", "Title", [], {})
        cache.clear()

        assert len(cache) == 0
        assert cache.get("https://example.com") is None

    def test_cache_hit_count(self) -> None:
        """Test that cache hit count increments."""
        cache = LRUCache(max_size=5)

        cache.put("https://example.com", "content", "Title", [], {})

        cache.get("https://example.com")
        cache.get("https://example.com")
        cached = cache.get("https://example.com")

        assert cached is not None
        assert cached.hits == 3


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_rate_limiter_initial_state(self) -> None:
        """Test rate limiter initial state."""
        limiter = RateLimiter(requests_per_second=10.0)

        assert limiter.requests_per_second == 10.0
        assert limiter.min_interval == 0.1

    def test_rate_limiter_wait(self) -> None:
        """Test that rate limiter enforces delays."""
        limiter = RateLimiter(requests_per_second=100.0)  # 10ms between requests

        # First request should not wait
        start = time.time()
        limiter.wait()
        first_wait = time.time() - start
        assert first_wait < 0.05  # Should be nearly instant

        # Second request should wait at least min_interval
        start = time.time()
        limiter.wait()
        second_wait = time.time() - start
        assert second_wait >= 0.009  # At least ~10ms

    def test_rate_limiter_reset(self) -> None:
        """Test rate limiter reset."""
        limiter = RateLimiter(requests_per_second=10.0)
        limiter.wait()

        limiter.reset()

        # After reset, next wait should be instant
        start = time.time()
        limiter.wait()
        wait_time = time.time() - start
        assert wait_time < 0.05


class TestBrowserResult:
    """Tests for the BrowserResult dataclass."""

    def test_browser_result_success(self) -> None:
        """Test successful browser result."""
        result = BrowserResult(
            url="https://example.com",
            title="Test Title",
            content="Test content",
            links=["https://example.com/link1"],
            metadata={"key": "value"},
            success=True,
            status_code=200,
        )

        assert result.success is True
        assert result.error is None
        assert result.url == "https://example.com"
        assert result.title == "Test Title"

    def test_browser_result_error(self) -> None:
        """Test error browser result."""
        result = BrowserResult(
            url="https://example.com",
            title="",
            content="",
            success=False,
            error="Connection failed",
        )

        assert result.success is False
        assert result.error == "Connection failed"


class TestBrowserTool:
    """Tests for the BrowserTool class."""

    @pytest.fixture
    def browser_tool(self) -> BrowserTool:
        """Create a BrowserTool instance for testing."""
        return BrowserTool(cache_size=10, requests_per_second=10.0)

    def test_browser_tool_initialization(self, browser_tool: BrowserTool) -> None:
        """Test BrowserTool initialization."""
        assert browser_tool.name == "browser"
        assert browser_tool._cache.max_size == 10
        assert browser_tool._rate_limiter.requests_per_second == 10.0

    def test_is_valid_url_valid(self, browser_tool: BrowserTool) -> None:
        """Test URL validation with valid URLs."""
        assert browser_tool._is_valid_url("https://example.com") is True
        assert browser_tool._is_valid_url("http://example.com/path") is True
        assert browser_tool._is_valid_url("https://www.example.com?q=test") is True

    def test_is_valid_url_invalid(self, browser_tool: BrowserTool) -> None:
        """Test URL validation with invalid URLs."""
        assert browser_tool._is_valid_url("not-a-url") is False
        assert browser_tool._is_valid_url("ftp://example.com") is False
        assert browser_tool._is_valid_url("") is False

    @patch("src.tools.browser_tool.requests.Session.get")
    def test_visit_success(
        self, mock_get: Mock, browser_tool: BrowserTool
    ) -> None:
        """Test successful URL visit."""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Heading</h1>
                <p>This is test content.</p>
                <a href="https://example.com/link1">Link 1</a>
            </body>
        </html>
        """
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.url = "https://example.com"
        mock_get.return_value = mock_response

        result = browser_tool.visit("https://example.com")

        assert result.success is True
        assert result.url == "https://example.com"
        assert result.title == "Test Page"
        assert "Test Heading" in result.content or "test content" in result.content.lower()
        assert result.status_code == 200

    @patch("src.tools.browser_tool.requests.Session.get")
    def test_visit_timeout(
        self, mock_get: Mock, browser_tool: BrowserTool
    ) -> None:
        """Test URL visit with timeout."""
        mock_get.side_effect = requests.exceptions.Timeout()

        result = browser_tool.visit("https://example.com")

        assert result.success is False
        assert result.error == "Request timed out"

    @patch("src.tools.browser_tool.requests.Session.get")
    def test_visit_connection_error(
        self, mock_get: Mock, browser_tool: BrowserTool
    ) -> None:
        """Test URL visit with connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        result = browser_tool.visit("https://example.com")

        assert result.success is False
        assert "Connection failed" in result.error

    def test_visit_invalid_url(self, browser_tool: BrowserTool) -> None:
        """Test visiting an invalid URL."""
        result = browser_tool.visit("not-a-valid-url")

        assert result.success is False
        assert "Invalid URL" in result.error

    @patch("src.tools.browser_tool.requests.Session.get")
    def test_visit_caching(
        self, mock_get: Mock, browser_tool: BrowserTool
    ) -> None:
        """Test that visiting the same URL uses cache."""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html><head><title>Test</title></head><body>Content</body></html>"
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.url = "https://example.com"
        mock_get.return_value = mock_response

        # First visit
        result1 = browser_tool.visit("https://example.com")
        assert result1.from_cache is False

        # Second visit should use cache
        result2 = browser_tool.visit("https://example.com")
        assert result2.from_cache is True

        # Should only have made one HTTP request
        assert mock_get.call_count == 1

    def test_clear_cache(self, browser_tool: BrowserTool) -> None:
        """Test cache clearing."""
        browser_tool._cache.put("https://example.com", "content", "Title", [], {})

        browser_tool.clear_cache()

        assert len(browser_tool._cache) == 0

    def test_get_cache_stats(self, browser_tool: BrowserTool) -> None:
        """Test cache statistics."""
        browser_tool._cache.put("https://example.com", "content", "Title", [], {})
        browser_tool._cache.get("https://example.com")  # Increment hit count

        stats = browser_tool.get_cache_stats()

        assert stats["cached_urls"] == 1
        assert stats["max_size"] == 10
        assert stats["total_hits"] >= 1

    @patch("src.tools.browser_tool.DuckDuckGoSearchAPIWrapper")
    def test_search(
        self, mock_search_wrapper: Mock, browser_tool: BrowserTool
    ) -> None:
        """Test web search."""
        # Mock the search API
        mock_search = Mock()
        mock_search.results.return_value = {
            "results": [
                {
                    "title": "Test Result",
                    "link": "https://example.com",
                    "snippet": "Test snippet",
                }
            ]
        }
        mock_search_wrapper.return_value = mock_search

        result = browser_tool.search("test query")

        assert "Test Result" in result
        assert "https://example.com" in result

    @patch("src.tools.browser_tool.requests.Session.get")
    def test_extract_content(
        self, mock_get: Mock, browser_tool: BrowserTool
    ) -> None:
        """Test content extraction."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html>
            <head><title>Article Title</title></head>
            <body>
                <article>
                    <p>This is the main article content.</p>
                    <p>More content here.</p>
                </article>
            </body>
        </html>
        """
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.url = "https://example.com/article"
        mock_get.return_value = mock_response

        result = browser_tool.extract_content("https://example.com/article")

        assert result.success is True
        assert result.title == "Article Title"

    @patch("src.tools.browser_tool.requests.Session.get")
    def test_get_links(
        self, mock_get: Mock, browser_tool: BrowserTool
    ) -> None:
        """Test link extraction."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html>
            <head><title>Links Page</title></head>
            <body>
                <a href="https://example.com/page1">Page 1</a>
                <a href="https://example.com/page2">Page 2</a>
                <a href="/relative/path">Relative</a>
                <a href="javascript:void(0)">JS Link</a>
                <a href="mailto:test@example.com">Email</a>
            </body>
        </html>
        """
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.url = "https://example.com"
        mock_get.return_value = mock_response

        result = browser_tool.get_links("https://example.com")

        assert result.success is True
        assert len(result.links) >= 2  # At least the valid links
        assert "https://example.com/page1" in result.links
        assert "https://example.com/page2" in result.links

    def test_browse_url_alias(self, browser_tool: BrowserTool) -> None:
        """Test that browse_url is an alias for visit."""
        # Just verify the method exists and calls visit
        assert hasattr(browser_tool, "browse_url")
        assert browser_tool.browse_url == browser_tool.visit

    @patch("src.tools.browser_tool.requests.Session.get")
    def test_scroll_simulation(
        self, mock_get: Mock, browser_tool: BrowserTool
    ) -> None:
        """Test scroll simulation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html><head><title>Scroll Page</title></head><body>Content</body></html>"
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.url = "https://example.com"
        mock_get.return_value = mock_response

        result = browser_tool.scroll("https://example.com", scroll_times=5)

        assert result.success is True
        assert result.metadata.get("scroll_simulated") is True
        assert result.metadata.get("scroll_times") == 5

    def test_run_with_action(self, browser_tool: BrowserTool) -> None:
        """Test the _run method with different actions."""
        # Test with visit action (default)
        with patch.object(browser_tool, "visit") as mock_visit:
            mock_visit.return_value = BrowserResult(
                url="https://example.com",
                title="Test",
                content="Content",
                success=True,
            )
            result = browser_tool._run("https://example.com", action="visit")
            assert mock_visit.called

        # Test with search action
        with patch.object(browser_tool, "search") as mock_search:
            mock_search.return_value = "Search results"
            result = browser_tool._run("test query", action="search")
            assert mock_search.called
            assert result == "Search results"

    def test_run_invalid_action(self, browser_tool: BrowserTool) -> None:
        """Test _run with invalid action defaults to visit."""
        with patch.object(browser_tool, "visit") as mock_visit:
            mock_visit.return_value = BrowserResult(
                url="https://example.com",
                title="Test",
                content="Content",
                success=True,
            )
            # Invalid action should default to visit
            result = browser_tool._run("https://example.com", action="invalid")
            assert mock_visit.called


class TestBrowserToolIntegration:
    """Integration tests for BrowserTool."""

    def test_full_browsing_workflow(self) -> None:
        """Test a complete browsing workflow."""
        tool = BrowserTool(cache_size=5, requests_per_second=5.0)

        # Verify tool has all expected methods
        assert hasattr(tool, "visit")
        assert hasattr(tool, "search")
        assert hasattr(tool, "extract_content")
        assert hasattr(tool, "get_links")
        assert hasattr(tool, "scroll")
        assert hasattr(tool, "browse_url")
        assert hasattr(tool, "clear_cache")
        assert hasattr(tool, "get_cache_stats")

    def test_tool_schema(self) -> None:
        """Test the tool's input schema."""
        tool = BrowserTool()

        assert tool.args_schema is not None
        # Verify schema has required fields
        schema = tool.args_schema
        assert hasattr(schema, "model_fields")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
