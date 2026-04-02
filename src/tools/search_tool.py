"""
Search tool for the Research Assistant.

This module provides a tool for searching the web and retrieving
relevant information for research tasks.
"""

from typing import Any, Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils.config import get_settings
from src.utils.helpers import sanitize_text, truncate_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SearchInput(BaseModel):
    """Input schema for the SearchTool."""

    query: str = Field(description="The search query to execute")
    num_results: Optional[int] = Field(
        default=None,
        description="Number of results to return (default: from config)",
    )


class SearchResult(BaseModel):
    """Represents a single search result."""

    title: str
    url: str
    snippet: str
    source: str


class SearchTool(BaseTool):
    """
    A tool for performing web searches to gather research information.

    This tool supports multiple search engines and returns structured
    search results that can be used by other components of the system.

    Attributes:
        name: The name of the tool.
        description: A description of what the tool does.
        args_schema: The schema for tool input validation.

    Example:
        >>> tool = SearchTool()
        >>> results = tool.run({"query": "latest AI developments"})
        >>> print(results)
    """

    name: str = "search"
    description: str = (
        "Search the web for information on a given topic. "
        "Use this tool to find current information, news, and resources. "
        "Input should be a search query string."
    )
    args_schema: Type[BaseModel] = SearchInput

    _settings: Any = Field(default_factory=get_settings)
    _search_engine: Optional[Any] = Field(default=None, exclude=True)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the SearchTool with configuration."""
        super().__init__(**kwargs)
        self._initialize_search_engine()

    def _initialize_search_engine(self) -> None:
        """Initialize the search engine based on configuration."""
        engine = self._settings.search_engine.lower()

        if engine == "duckduckgo":
            try:
                from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

                self._search_engine = DuckDuckGoSearchAPIWrapper(
                    max_results=self._settings.max_search_results
                )
                logger.info("Initialized DuckDuckGo search engine")
            except ImportError:
                logger.warning("DuckDuckGo search not available, using mock search")
                self._search_engine = None
        else:
            logger.warning(f"Unknown search engine: {engine}, using mock search")
            self._search_engine = None

    def _mock_search(self, query: str, num_results: int) -> list[dict[str, str]]:
        """
        Perform a mock search when no search engine is available.

        Args:
            query: The search query.
            num_results: Number of results to return.

        Returns:
            A list of mock search results.
        """
        return [
            {
                "title": f"Search Result {i + 1} for: {query}",
                "link": f"https://example.com/result/{i + 1}",
                "snippet": f"This is a mock search result for the query '{query}'. "
                f"In production, this would contain actual search results.",
            }
            for i in range(min(num_results, 3))
        ]

    def _run(
        self,
        query: str,
        num_results: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Execute the search and return formatted results.

        Args:
            query: The search query string.
            num_results: Optional override for number of results.
            run_manager: Optional callback manager for tool execution.

        Returns:
            A formatted string containing search results.
        """
        num_results = num_results or self._settings.max_search_results
        sanitized_query = sanitize_text(query)

        logger.info(f"Executing search for: {sanitized_query}")

        try:
            if self._search_engine:
                results = self._search_engine.results(
                    sanitized_query, num_results=num_results
                )
            else:
                results = {"results": self._mock_search(sanitized_query, num_results)}

            formatted_results = self._format_results(results)
            logger.info(f"Search completed with {len(results.get('results', []))} results")

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return f"Search failed: {str(e)}"

    async def _arun(
        self,
        query: str,
        num_results: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        Async version of the search execution.

        Args:
            query: The search query string.
            num_results: Optional override for number of results.
            run_manager: Optional callback manager for async tool execution.

        Returns:
            A formatted string containing search results.
        """
        # For now, delegate to sync version
        return self._run(query, num_results, run_manager=run_manager.get_sync() if run_manager else None)

    def _format_results(self, results: dict[str, Any]) -> str:
        """
        Format search results into a readable string.

        Args:
            results: Raw search results from the search engine.

        Returns:
            A formatted string with search results.
        """
        formatted = []
        search_results = results.get("results", [])

        for i, result in enumerate(search_results, 1):
            title = result.get("title", "No Title")
            url = result.get("link", result.get("url", "No URL"))
            snippet = truncate_text(result.get("snippet", result.get("content", "")), max_length=300)

            formatted.append(f"[{i}] {title}\n    URL: {url}\n    Summary: {snippet}\n")

        if not formatted:
            return "No results found for the query."

        return "\n".join(formatted)

    def search_with_metadata(
        self, query: str, num_results: Optional[int] = None
    ) -> list[SearchResult]:
        """
        Perform a search and return structured SearchResult objects.

        Args:
            query: The search query.
            num_results: Optional number of results to return.

        Returns:
            A list of SearchResult objects.
        """
        num_results = num_results or self._settings.max_search_results
        sanitized_query = sanitize_text(query)

        try:
            if self._search_engine:
                results = self._search_engine.results(
                    sanitized_query, num_results=num_results
                )
            else:
                results = {"results": self._mock_search(sanitized_query, num_results)}

            return [
                SearchResult(
                    title=r.get("title", "No Title"),
                    url=r.get("link", r.get("url", "No URL")),
                    snippet=r.get("snippet", r.get("content", "")),
                    source=self._settings.search_engine,
                )
                for r in results.get("results", [])
            ]

        except Exception as e:
            logger.error(f"Search with metadata failed: {str(e)}")
            return []
