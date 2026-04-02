"""
Citation tool for the Research Assistant.

This module provides a tool for managing and formatting citations
in research reports and documents.
"""

import hashlib
from datetime import datetime
from typing import Any, Optional, Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils.helpers import extract_urls, format_timestamp, generate_id
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CitationInput(BaseModel):
    """Input schema for the CiteTool."""

    content: str = Field(description="The content or source to cite")
    citation_type: Optional[str] = Field(
        default="web",
        description="Type of citation: web, article, book, or custom",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the citation (author, title, date, etc.)",
    )


class Citation(BaseModel):
    """Represents a citation entry."""

    id: str
    content: str
    citation_type: str
    url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    accessed_date: str
    citation_key: str

    def format_apa(self) -> str:
        """Format the citation in APA style."""
        parts = []
        if self.author:
            parts.append(self.author)
        if self.date:
            parts.append(f"({self.date})")
        if self.title:
            parts.append(f"{self.title}.")
        if self.url:
            parts.append(f"Retrieved from {self.url}")

        return " ".join(parts) if parts else self.content

    def format_mla(self) -> str:
        """Format the citation in MLA style."""
        parts = []
        if self.author:
            parts.append(self.author)
        if self.title:
            parts.append(f'"{self.title}."')
        if self.date:
            parts.append(self.date)
        if self.url:
            parts.append(self.url)

        return ". ".join(parts) if parts else self.content

    def format_chicago(self) -> str:
        """Format the citation in Chicago style."""
        parts = []
        if self.author:
            parts.append(self.author)
        if self.title:
            parts.append(f'"{self.title}."')
        if self.date:
            parts.append(f"Last modified {self.date}")
        if self.url:
            parts.append(self.url)

        return ". ".join(parts) if parts else self.content


class CitationManager:
    """
    Manages a collection of citations for a research session.

    This class handles adding, retrieving, and formatting citations
    throughout the research process.
    """

    def __init__(self) -> None:
        """Initialize the CitationManager with an empty citation store."""
        self._citations: dict[str, Citation] = {}
        self._citation_order: list[str] = []

    def add_citation(self, citation: Citation) -> str:
        """
        Add a citation to the collection.

        Args:
            citation: The Citation object to add.

        Returns:
            The citation key for reference.
        """
        self._citations[citation.id] = citation
        self._citation_order.append(citation.id)
        return citation.citation_key

    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """
        Retrieve a citation by its ID.

        Args:
            citation_id: The ID of the citation to retrieve.

        Returns:
            The Citation object or None if not found.
        """
        return self._citations.get(citation_id)

    def get_all_citations(self) -> list[Citation]:
        """
        Get all citations in order of addition.

        Returns:
            A list of all Citation objects.
        """
        return [self._citations[cid] for cid in self._citation_order]

    def format_bibliography(
        self, style: str = "apa", citation_ids: Optional[list[str]] = None
    ) -> str:
        """
        Format a bibliography from the stored citations.

        Args:
            style: The citation style (apa, mla, chicago).
            citation_ids: Optional list of specific citation IDs to include.

        Returns:
            A formatted bibliography string.
        """
        if citation_ids:
            citations = [
                self._citations[cid]
                for cid in citation_ids
                if cid in self._citations
            ]
        else:
            citations = self.get_all_citations()

        if not citations:
            return "No citations available."

        style_methods = {
            "apa": lambda c: c.format_apa(),
            "mla": lambda c: c.format_mla(),
            "chicago": lambda c: c.format_chicago(),
        }

        format_func = style_methods.get(style.lower(), style_methods["apa"])

        bibliography = []
        for i, citation in enumerate(citations, 1):
            formatted = format_func(citation)
            bibliography.append(f"[{i}] {formatted}")

        return "\n".join(bibliography)

    def clear(self) -> None:
        """Clear all stored citations."""
        self._citations.clear()
        self._citation_order.clear()


class CiteTool(BaseTool):
    """
    A tool for managing and formatting citations in research.

    This tool helps track sources used during research and generates
    properly formatted citations in various styles.

    Attributes:
        name: The name of the tool.
        description: A description of what the tool does.
        args_schema: The schema for tool input validation.
        citation_manager: The manager for storing citations.

    Example:
        >>> tool = CiteTool()
        >>> citation = tool.run({
        ...     "content": "https://example.com/article",
        ...     "citation_type": "web"
        ... })
    """

    name: str = "cite"
    description: str = (
        "Create and manage citations for research sources. "
        "Use this tool to track sources, generate citations in various formats, "
        "and build bibliographies. Input should be the source content or URL."
    )
    args_schema: Type[BaseModel] = CitationInput

    citation_manager: CitationManager = Field(default_factory=CitationManager)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the CiteTool."""
        super().__init__(**kwargs)

    def _run(
        self,
        content: str,
        citation_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Create a citation and return the formatted citation string.

        Args:
            content: The content or source to cite.
            citation_type: Type of citation (web, article, book, custom).
            metadata: Additional metadata for the citation.
            run_manager: Optional callback manager for tool execution.

        Returns:
            A formatted citation string with the citation key.
        """
        citation_type = citation_type or "web"
        metadata = metadata or {}

        logger.info(f"Creating {citation_type} citation for content")

        try:
            citation = self._create_citation(content, citation_type, metadata)
            citation_key = self.citation_manager.add_citation(citation)

            formatted = citation.format_apa()
            result = f"[{citation_key}] {formatted}"

            logger.info(f"Citation created with key: {citation_key}")
            return result

        except Exception as e:
            logger.error(f"Citation creation failed: {str(e)}")
            return f"Citation failed: {str(e)}"

    async def _arun(
        self,
        content: str,
        citation_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        Async version of citation creation.

        Args:
            content: The content or source to cite.
            citation_type: Type of citation.
            metadata: Additional metadata.
            run_manager: Optional callback manager for async execution.

        Returns:
            A formatted citation string.
        """
        # Citation creation is synchronous, so delegate to _run
        return self._run(
            content, citation_type, metadata, run_manager=run_manager.get_sync() if run_manager else None
        )

    def _create_citation(
        self, content: str, citation_type: str, metadata: dict[str, Any]
    ) -> Citation:
        """
        Create a Citation object from the provided information.

        Args:
            content: The content or source.
            citation_type: The type of citation.
            metadata: Additional metadata.

        Returns:
            A Citation object.
        """
        # Extract URL if present in content
        urls = extract_urls(content)
        url = urls[0] if urls else (metadata.get("url") if metadata else None)

        # Generate citation key from content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        citation_key = f"{citation_type}_{content_hash}"

        citation = Citation(
            id=generate_id(),
            content=content,
            citation_type=citation_type,
            url=url,
            title=metadata.get("title"),
            author=metadata.get("author"),
            date=metadata.get("date"),
            accessed_date=format_timestamp(),
            citation_key=citation_key,
        )

        return citation

    def get_bibliography(self, style: str = "apa") -> str:
        """
        Get a formatted bibliography of all citations.

        Args:
            style: The citation style (apa, mla, chicago).

        Returns:
            A formatted bibliography string.
        """
        return self.citation_manager.format_bibliography(style)

    def get_citation_count(self) -> int:
        """
        Get the number of stored citations.

        Returns:
            The count of citations.
        """
        return len(self.citation_manager._citations)

    def clear_citations(self) -> None:
        """Clear all stored citations."""
        self.citation_manager.clear()
        logger.info("All citations cleared")

    def cite_multiple(
        self, sources: list[dict[str, Any]]
    ) -> list[str]:
        """
        Create multiple citations at once.

        Args:
            sources: A list of source dictionaries with keys:
                - content: The source content
                - citation_type: Optional type (default: "web")
                - metadata: Optional metadata dict

        Returns:
            A list of formatted citation strings.
        """
        results = []
        for source in sources:
            result = self._run(
                content=source.get("content", ""),
                citation_type=source.get("citation_type"),
                metadata=source.get("metadata"),
            )
            results.append(result)
        return results
