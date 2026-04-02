"""
Summarize tool for the Research Assistant.

This module provides a tool for summarizing text content using
LLM-based summarization techniques.
"""

from typing import Any, Optional, Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.config import get_settings
from src.utils.helpers import chunk_text, sanitize_text, truncate_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SummarizeInput(BaseModel):
    """Input schema for the SummarizeTool."""

    text: str = Field(description="The text content to summarize")
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum length of the summary in words (default: 200)",
    )
    style: Optional[str] = Field(
        default="concise",
        description="Summary style: concise, detailed, bullet_points, or executive",
    )


class SummarizeTool(BaseTool):
    """
    A tool for summarizing text content using LLM-based techniques.

    This tool can handle long documents by chunking and provides
    multiple summarization styles for different use cases.

    Attributes:
        name: The name of the tool.
        description: A description of what the tool does.
        args_schema: The schema for tool input validation.
        llm: The language model to use for summarization.

    Example:
        >>> tool = SummarizeTool(llm=my_llm)
        >>> summary = tool.run({"text": "Long article content..."})
    """

    name: str = "summarize"
    description: str = (
        "Summarize a piece of text into a concise summary. "
        "Use this tool to condense long articles, documents, or research papers. "
        "Input should be the text content to summarize."
    )
    args_schema: Type[BaseModel] = SummarizeInput

    _settings: Any = Field(default_factory=get_settings)
    llm: Optional[BaseLanguageModel] = Field(default=None)
    _summarize_prompt: ChatPromptTemplate = Field(default=None, exclude=True)

    def __init__(self, llm: Optional[BaseLanguageModel] = None, **kwargs: Any) -> None:
        """
        Initialize the SummarizeTool.

        Args:
            llm: The language model to use for summarization.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(llm=llm, **kwargs)
        self._initialize_prompt()

    def _initialize_prompt(self) -> None:
        """Initialize the summarization prompt template."""
        self._summarize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert research assistant skilled at summarizing "
                    "complex information clearly and accurately.\n"
                    "Provide a {style} summary of the given text.\n"
                    "Focus on key findings, main arguments, and important conclusions.\n"
                    "Do not add information not present in the original text.",
                ),
                ("human", "Text to summarize:\n\n{text}"),
            ]
        )

    def _run(
        self,
        text: str,
        max_length: Optional[int] = None,
        style: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Execute the summarization and return the summary.

        Args:
            text: The text content to summarize.
            max_length: Optional maximum length in words.
            style: Optional summary style (concise, detailed, bullet_points, executive).
            run_manager: Optional callback manager for tool execution.

        Returns:
            The generated summary as a string.
        """
        sanitized_text = sanitize_text(text)
        max_length = max_length or 200
        style = style or "concise"

        logger.info(f"Summarizing text ({len(sanitized_text)} chars) with style: {style}")

        try:
            # Check if text needs chunking
            if len(sanitized_text) > 4000:
                summary = self._summarize_chunked(sanitized_text, style, max_length)
            else:
                summary = self._summarize_single(sanitized_text, style, max_length)

            logger.info(f"Summary generated ({len(summary)} chars)")
            return summary

        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return f"Summarization failed: {str(e)}"

    async def _arun(
        self,
        text: str,
        max_length: Optional[int] = None,
        style: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        Async version of the summarization execution.

        Args:
            text: The text content to summarize.
            max_length: Optional maximum length in words.
            style: Optional summary style.
            run_manager: Optional callback manager for async tool execution.

        Returns:
            The generated summary as a string.
        """
        if self.llm is None:
            return self._run(text, max_length, style, run_manager=run_manager.get_sync() if run_manager else None)

        sanitized_text = sanitize_text(text)
        max_length = max_length or 200
        style = style or "concise"

        try:
            prompt = self._summarize_prompt.format(text=sanitized_text, style=style)

            if len(sanitized_text) > 4000:
                summary = await self._summarize_chunked_async(
                    sanitized_text, style, max_length
                )
            else:
                response = await self.llm.ainvoke(prompt)
                summary = response.content if hasattr(response, "content") else str(response)

            return summary

        except Exception as e:
            logger.error(f"Async summarization failed: {str(e)}")
            return f"Summarization failed: {str(e)}"

    def _summarize_single(
        self, text: str, style: str, max_length: int
    ) -> str:
        """
        Summarize a single chunk of text.

        Args:
            text: The text to summarize.
            style: The summary style.
            max_length: Maximum length in words.

        Returns:
            The summary string.
        """
        if self.llm is None:
            return self._extractive_summary(text, max_length)

        prompt = self._summarize_prompt.format(text=text, style=style)
        response = self.llm.invoke(prompt)

        summary = response.content if hasattr(response, "content") else str(response)
        return truncate_text(summary, max_length=max_length * 6)  # Approx chars

    async def _summarize_chunked_async(
        self, text: str, style: str, max_length: int
    ) -> str:
        """
        Summarize text by chunking and combining summaries asynchronously.

        Args:
            text: The text to summarize.
            style: The summary style.
            max_length: Maximum length in words.

        Returns:
            The combined summary string.
        """
        chunks = chunk_text(text, chunk_size=3500, overlap=200)
        chunk_summaries = []

        for chunk in chunks:
            prompt = self._summarize_prompt.format(text=chunk, style=style)
            response = await self.llm.ainvoke(prompt)
            summary = response.content if hasattr(response, "content") else str(response)
            chunk_summaries.append(summary)

        # Combine and summarize the chunk summaries
        combined = "\n\n".join(chunk_summaries)
        return self._summarize_single(combined, style, max_length)

    def _summarize_chunked(
        self, text: str, style: str, max_length: int
    ) -> str:
        """
        Summarize text by chunking and combining summaries.

        Args:
            text: The text to summarize.
            style: The summary style.
            max_length: Maximum length in words.

        Returns:
            The combined summary string.
        """
        chunks = chunk_text(text, chunk_size=3500, overlap=200)
        chunk_summaries = []

        for chunk in chunks:
            summary = self._summarize_single(chunk, style, max_length)
            chunk_summaries.append(summary)

        # Combine and summarize the chunk summaries
        combined = "\n\n".join(chunk_summaries)
        return self._summarize_single(combined, style, max_length)

    def _extractive_summary(self, text: str, max_length: int) -> str:
        """
        Generate an extractive summary when no LLM is available.

        This method extracts key sentences from the text as a fallback
        when no language model is configured.

        Args:
            text: The text to summarize.
            max_length: Maximum length in words.

        Returns:
            An extractive summary.
        """
        sentences = text.replace("\n", " ").split(". ")
        # Take first few sentences as a simple extractive summary
        summary_sentences = sentences[: min(5, len(sentences))]
        summary = ". ".join(summary_sentences)
        if summary and not summary.endswith("."):
            summary += "."
        return truncate_text(summary, max_length=max_length * 6)

    def set_llm(self, llm: BaseLanguageModel) -> None:
        """
        Set the language model for summarization.

        Args:
            llm: The language model to use.
        """
        self.llm = llm
        logger.info("LLM set for SummarizeTool")
