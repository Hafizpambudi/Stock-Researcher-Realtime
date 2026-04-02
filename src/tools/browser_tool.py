"""
Browser tool for the Research Assistant.

This module provides a comprehensive web browsing tool for the Research Assistant,
enabling navigation, content extraction, and link discovery with rate limiting,
caching, and error handling.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Type
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from readability import Document as ReadabilityDocument

from src.utils.config import get_settings
from src.utils.helpers import sanitize_text, truncate_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CachedResponse:
    """Represents a cached HTTP response."""

    url: str
    content: str
    title: str
    links: list[str]
    metadata: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    hits: int = 0


class LRUCache:
    """
    Least Recently Used (LRU) cache for storing fetched URLs.

    This cache helps avoid redundant HTTP requests by storing previously
    fetched page content with a configurable maximum size.

    Attributes:
        max_size: Maximum number of entries in the cache.
    """

    def __init__(self, max_size: int = 100) -> None:
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries to store.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, CachedResponse] = OrderedDict()

    def get(self, url: str) -> Optional[CachedResponse]:
        """
        Retrieve a cached response by URL.

        Args:
            url: The URL to look up.

        Returns:
            The cached response or None if not found.
        """
        if url in self._cache:
            cached = self._cache[url]
            cached.hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(url)
            return cached
        return None

    def put(
        self,
        url: str,
        content: str,
        title: str,
        links: list[str],
        metadata: dict[str, Any],
    ) -> None:
        """
        Store a response in the cache.

        Args:
            url: The URL of the response.
            content: The page content.
            title: The page title.
            links: List of links found on the page.
            metadata: Additional metadata about the page.
        """
        if url in self._cache:
            # Update existing entry
            self._cache[url] = CachedResponse(
                url=url,
                content=content,
                title=title,
                links=links,
                metadata=metadata,
            )
            self._cache.move_to_end(url)
        else:
            # Add new entry
            if len(self._cache) >= self.max_size:
                # Remove least recently used
                self._cache.popitem(last=False)

            self._cache[url] = CachedResponse(
                url=url,
                content=content,
                title=title,
                links=links,
                metadata=metadata,
            )

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)


@dataclass
class BrowserResult:
    """Represents the result of a browser operation."""

    url: str
    title: str
    content: str
    links: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    status_code: Optional[int] = None
    fetch_time: float = 0.0
    from_cache: bool = False


class BrowserTool(BaseTool):
    """
    A comprehensive web browsing tool for research tasks.

    This tool provides capabilities for:
    - Navigating to URLs and extracting content
    - Searching the web
    - Extracting main content from webpages (removing ads, navigation, etc.)
    - Getting all links from a page
    - Simulating scrolling for dynamic content
    - Rate limiting and caching for efficient browsing

    Attributes:
        name: The name of the tool.
        description: A description of what the tool does.
        args_schema: The schema for tool input validation.

    Example:
        >>> tool = BrowserTool()
        >>> result = tool.visit("https://example.com")
        >>> print(result.content)
    """

    name: str = "browser"
    description: str = (
        "Browse and extract content from web pages. Use this tool to visit URLs, "
        "extract main content, get links from pages, and search the web. "
        "Input should be a URL or search query."
    )
    args_schema: Type[BaseModel] = Field(default_factory=lambda: BrowserInput)

    _settings: Any = PrivateAttr(default_factory=get_settings)
    _session: requests.Session = PrivateAttr(default_factory=requests.Session)
    _cache: LRUCache = PrivateAttr(default_factory=lambda: LRUCache(max_size=100))
    _rate_limiter: Any = PrivateAttr(
        default_factory=lambda: RateLimiter(requests_per_second=2)
    )

    def __init__(
        self,
        cache_size: int = 100,
        requests_per_second: float = 2.0,
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the BrowserTool.

        Args:
            cache_size: Maximum number of URLs to cache.
            requests_per_second: Maximum requests per second for rate limiting.
            timeout: Request timeout in seconds.
            **kwargs: Additional keyword arguments passed to BaseTool.
        """
        super().__init__(**kwargs)
        self._cache = LRUCache(max_size=cache_size)
        self._rate_limiter = RateLimiter(requests_per_second=requests_per_second)
        self._timeout = timeout
        self._setup_session()

    def _setup_session(self) -> None:
        """Configure the requests session with appropriate headers."""
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

    def _run(
        self,
        url: str,
        action: str = "visit",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Execute the browser tool.

        Args:
            url: The URL to visit or search query.
            action: The action to perform (visit, search, extract, links).
            run_manager: Optional callback manager for tool execution.

        Returns:
            A formatted string with the browsing results.
        """
        action = action.lower()

        if action == "search":
            return self.search(url)
        elif action == "extract":
            result = self.extract_content(url)
            return self._format_result(result)
        elif action == "links":
            result = self.get_links(url)
            return self._format_links_result(result)
        else:
            result = self.visit(url)
            return self._format_result(result)

    async def _arun(
        self,
        url: str,
        action: str = "visit",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        Async version of the browser tool execution.

        Args:
            url: The URL to visit or search query.
            action: The action to perform.
            run_manager: Optional callback manager for async execution.

        Returns:
            A formatted string with the browsing results.
        """
        # For now, delegate to sync version
        sync_manager = run_manager.get_sync() if run_manager else None
        return self._run(url, action, run_manager=sync_manager)

    def visit(self, url: str) -> BrowserResult:
        """
        Navigate to a URL and extract its content.

        Args:
            url: The URL to visit.

        Returns:
            A BrowserResult containing the page content, title, and links.
        """
        start_time = time.time()
        sanitized_url = sanitize_text(url)

        # Validate URL
        if not self._is_valid_url(sanitized_url):
            return BrowserResult(
                url=url,
                title="",
                content="",
                success=False,
                error=f"Invalid URL: {url}",
            )

        # Check cache
        cached = self._cache.get(sanitized_url)
        if cached:
            logger.debug(f"Cache hit for: {sanitized_url}")
            return BrowserResult(
                url=sanitized_url,
                title=cached.title,
                content=cached.content,
                links=cached.links,
                metadata=cached.metadata,
                from_cache=True,
                fetch_time=0.0,
            )

        # Rate limiting
        self._rate_limiter.wait()

        logger.info(f"Visiting URL: {sanitized_url}")

        try:
            response = self._session.get(sanitized_url, timeout=self._timeout)
            response.raise_for_status()

            # Parse the page
            soup = BeautifulSoup(response.content, "lxml")

            # Extract title
            title = self._extract_title(soup)

            # Extract main content using readability
            content = self._extract_main_content(response.content)

            # Extract links
            links = self._extract_links(soup, sanitized_url)

            # Extract metadata
            metadata = self._extract_metadata(soup, response)

            fetch_time = time.time() - start_time

            # Cache the result
            self._cache.put(
                url=sanitized_url,
                content=content,
                title=title,
                links=links[:50],  # Cache first 50 links
                metadata=metadata,
            )

            logger.info(f"Successfully fetched: {sanitized_url} in {fetch_time:.2f}s")

            return BrowserResult(
                url=sanitized_url,
                title=title,
                content=content,
                links=links,
                metadata=metadata,
                status_code=response.status_code,
                fetch_time=fetch_time,
            )

        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching: {sanitized_url}")
            return BrowserResult(
                url=sanitized_url,
                title="",
                content="",
                success=False,
                error="Request timed out",
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {sanitized_url}: {str(e)}")
            return BrowserResult(
                url=sanitized_url,
                title="",
                content="",
                success=False,
                error=str(e),
            )

    def search(self, query: str) -> str:
        """
        Search the web and return results with URLs.

        Args:
            query: The search query.

        Returns:
            A formatted string with search results.
        """
        sanitized_query = sanitize_text(query)
        logger.info(f"Searching for: {sanitized_query}")

        try:
            # Use DuckDuckGo for search (via langchain-community if available)
            from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

            search = DuckDuckGoSearchAPIWrapper(max_results=10)
            results = search.results(sanitized_query, num_results=10)

            if not results or "results" not in results:
                return "No search results found."

            formatted = []
            for i, result in enumerate(results["results"], 1):
                title = result.get("title", "No Title")
                url = result.get("link", result.get("url", "No URL"))
                snippet = truncate_text(
                    result.get("snippet", result.get("content", "")), max_length=200
                )
                formatted.append(f"[{i}] {title}\n    URL: {url}\n    Summary: {snippet}\n")

            logger.info(f"Search completed with {len(results['results'])} results")
            return "\n".join(formatted)

        except ImportError:
            logger.warning("DuckDuckGo search not available")
            return "Search functionality not available. Please install duckduckgo-search."
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return f"Search failed: {str(e)}"

    def extract_content(self, url: str) -> BrowserResult:
        """
        Extract main content from a webpage (remove ads, navigation, etc.).

        Args:
            url: The URL to extract content from.

        Returns:
            A BrowserResult with the extracted main content.
        """
        result = self.visit(url)

        if not result.success:
            return result

        # Re-extract content using readability for cleaner output
        try:
            response = self._session.get(url, timeout=self._timeout)
            doc = ReadabilityDocument(response.content)
            result.content = doc.summary()
            result.title = doc.title() or result.title
            logger.info(f"Content extracted from: {url}")
        except Exception as e:
            logger.warning(f"Readability extraction failed: {str(e)}")
            # Keep the original content

        return result

    def get_links(self, url: str) -> BrowserResult:
        """
        Get all links from a webpage.

        Args:
            url: The URL to extract links from.

        Returns:
            A BrowserResult with the list of links.
        """
        result = self.visit(url)

        if not result.success:
            return result

        logger.info(f"Extracted {len(result.links)} links from: {url}")
        return result

    def scroll(self, url: str, scroll_times: int = 3) -> BrowserResult:
        """
        Simulate scrolling for dynamic content (requires JavaScript).

        Note: This is a limited implementation since we're using requests.
        For full JavaScript support, consider using Selenium or Playwright.

        Args:
            url: The URL to scroll.
            scroll_times: Number of scroll attempts.

        Returns:
            A BrowserResult with the content after scrolling.
        """
        logger.info(f"Scroll simulation for: {url} ({scroll_times} times)")

        # Since we're using requests (not a browser), we can't actually scroll
        # This is a placeholder that just visits the page
        # For real scrolling, you'd need Selenium/Playwright
        result = self.visit(url)

        if result.success:
            result.metadata["scroll_simulated"] = True
            result.metadata["scroll_times"] = scroll_times

        return result

    def browse_url(self, url: str) -> BrowserResult:
        """
        Convenience method for browsing a URL (alias for visit).

        Args:
            url: The URL to browse.

        Returns:
            A BrowserResult with the page content.
        """
        return self.visit(url)

    def clear_cache(self) -> None:
        """Clear the URL cache."""
        self._cache.clear()
        logger.info("Browser cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            A dictionary with cache statistics.
        """
        total_hits = sum(c.hits for c in self._cache._cache.values())
        return {
            "cached_urls": len(self._cache),
            "max_size": self._cache.max_size,
            "total_hits": total_hits,
        }

    def _is_valid_url(self, url: str) -> bool:
        """
        Validate a URL.

        Args:
            url: The URL to validate.

        Returns:
            True if the URL is valid, False otherwise.
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in [
                "http",
                "https",
            ]
        except Exception:
            return False

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract the page title from BeautifulSoup.

        Args:
            soup: BeautifulSoup object.

        Returns:
            The page title.
        """
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return sanitize_text(title_tag.string)

        # Fallback to h1
        h1_tag = soup.find("h1")
        if h1_tag:
            return sanitize_text(h1_tag.get_text(strip=True))

        return "Untitled Page"

    def _extract_main_content(self, html_content: bytes) -> str:
        """
        Extract main content using readability-lxml.

        Args:
            html_content: Raw HTML content.

        Returns:
            The extracted main content.
        """
        try:
            doc = ReadabilityDocument(html_content)
            content = doc.summary()
            return sanitize_text(content)
        except Exception as e:
            logger.warning(f"Readability extraction failed: {str(e)}")
            # Fallback to basic extraction
            soup = BeautifulSoup(html_content, "lxml")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get text content
            text = soup.get_text(separator="\n", strip=True)
            return sanitize_text(text)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """
        Extract all links from a page.

        Args:
            soup: BeautifulSoup object.
            base_url: The base URL for resolving relative links.

        Returns:
            A list of absolute URLs.
        """
        links = []
        seen = set()

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()

            # Skip javascript, mailto, and anchor links
            if href.startswith(("javascript:", "mailto:", "#")):
                continue

            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)

            # Deduplicate
            if absolute_url not in seen and self._is_valid_url(absolute_url):
                links.append(absolute_url)
                seen.add(absolute_url)

        return links

    def _extract_metadata(
        self, soup: BeautifulSoup, response: requests.Response
    ) -> dict[str, Any]:
        """
        Extract metadata from a page.

        Args:
            soup: BeautifulSoup object.
            response: The HTTP response.

        Returns:
            A dictionary of metadata.
        """
        metadata = {
            "url": response.url,
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "fetched_at": datetime.now().isoformat(),
        }

        # Extract meta tags
        meta_tags = soup.find_all("meta")
        for meta in meta_tags:
            name = meta.get("name") or meta.get("property")
            content = meta.get("content")
            if name and content:
                metadata[f"meta_{name}"] = content

        # Extract Open Graph data
        og_tags = soup.find_all("meta", property=lambda x: x and x.startswith("og:"))
        for tag in og_tags:
            property_name = tag.get("property", "")
            content = tag.get("content", "")
            if property_name and content:
                metadata[property_name] = content

        return metadata

    def _format_result(self, result: BrowserResult) -> str:
        """
        Format a BrowserResult into a readable string.

        Args:
            result: The BrowserResult to format.

        Returns:
            A formatted string.
        """
        if not result.success:
            return f"Error: {result.error}"

        lines = [
            f"URL: {result.url}",
            f"Title: {result.title}",
            f"Status: {result.status_code}",
            f"From Cache: {result.from_cache}",
            "",
            "--- Content ---",
            truncate_text(result.content, max_length=3000),
        ]

        if result.links:
            lines.append("")
            lines.append(f"--- Links ({len(result.links)} found) ---")
            for link in result.links[:10]:
                lines.append(f"  - {link}")
            if len(result.links) > 10:
                lines.append(f"  ... and {len(result.links) - 10} more")

        return "\n".join(lines)

    def _format_links_result(self, result: BrowserResult) -> str:
        """
        Format a links result into a readable string.

        Args:
            result: The BrowserResult to format.

        Returns:
            A formatted string with links.
        """
        if not result.success:
            return f"Error: {result.error}"

        lines = [
            f"URL: {result.url}",
            f"Title: {result.title}",
            "",
            f"--- Links ({len(result.links)} found) ---",
        ]

        for i, link in enumerate(result.links, 1):
            lines.append(f"[{i}] {link}")

        return "\n".join(lines)


class BrowserInput(BaseModel):
    """Input schema for the BrowserTool."""

    url: str = Field(description="The URL to visit or search query")
    action: str = Field(
        default="visit",
        description="Action to perform: visit, search, extract, or links",
    )


class RateLimiter:
    """
    Rate limiter to avoid overwhelming servers.

    This class implements a simple rate limiting mechanism that ensures
    a maximum number of requests per second.

    Attributes:
        requests_per_second: Maximum requests allowed per second.
    """

    def __init__(self, requests_per_second: float = 2.0) -> None:
        """
        Initialize the rate limiter.

        Args:
            requests_per_second: Maximum requests per second.
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self._last_request: Optional[float] = None

    def wait(self) -> None:
        """
        Wait if necessary to respect rate limiting.

        This method blocks until enough time has passed since the
        last request to stay within the rate limit.
        """
        if self._last_request is not None:
            elapsed = time.time() - self._last_request
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)

        self._last_request = time.time()

    def reset(self) -> None:
        """Reset the rate limiter state."""
        self._last_request = None
