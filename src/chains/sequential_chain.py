"""
Sequential Chain implementation for the Research Assistant.

This module implements a sequential chain that processes input through
a series of steps in order, passing the output of each step to the next.
"""

from typing import Any, Optional, Sequence

from langchain.callbacks.manager import Callbacks
from langchain.chains.base import Chain
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResearchSequentialChain(Chain):
    """
    A sequential chain for processing research tasks through multiple steps.

    This chain executes a series of operations in sequence, where the output
    of one step becomes the input to the next. It's useful for multi-stage
    research workflows like: search -> analyze -> summarize -> format.

    Attributes:
        name: The name of the chain.
        steps: A list of Runnable steps to execute in sequence.
        llm: Optional language model for LLM-based steps.
        input_key: The key for the chain's input.
        output_key: The key for the chain's output.

    Example:
        >>> chain = ResearchSequentialChain(
        ...     steps=[search_step, analyze_step, summarize_step],
        ...     llm=my_llm
        ... )
        >>> result = chain.invoke({"query": "AI trends"})
    """

    name: str = "research_sequential_chain"
    input_key: str = "query"
    output_key: str = "result"

    steps: Sequence[Runnable] = Field(default_factory=list)
    llm: Optional[BaseLanguageModel] = Field(default=None)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        steps: Optional[Sequence[Runnable]] = None,
        llm: Optional[BaseLanguageModel] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Sequential Chain.

        Args:
            steps: Optional list of Runnable steps to execute.
            llm: Optional language model for LLM-based operations.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(steps=steps or [], llm=llm, **kwargs)

    @property
    def input_keys(self) -> list[str]:
        """Return the expected input keys for the chain."""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return the output keys produced by the chain."""
        return [self.output_key]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[Callbacks] = None,
    ) -> dict[str, Any]:
        """
        Execute the sequential chain with the given inputs.

        Args:
            inputs: Dictionary containing the input values.
            run_manager: Optional callback manager for execution tracking.

        Returns:
            Dictionary containing the chain output.

        Raises:
            ValueError: If no steps are configured.
        """
        if not self.steps:
            raise ValueError("No steps configured for sequential chain")

        query = inputs.get(self.input_key, "")
        logger.info(f"Executing sequential chain with {len(self.steps)} steps")

        # Build the chain
        chain = self._build_chain()

        try:
            # Execute the chain
            result = chain.invoke({self.input_key: query}, config={"callbacks": run_manager})

            # Extract the final result
            if isinstance(result, dict):
                final_result = result.get("result", result.get(self.output_key, str(result)))
            else:
                final_result = str(result)

            logger.info("Sequential chain execution completed")
            return {self.output_key: final_result}

        except Exception as e:
            logger.error(f"Sequential chain execution failed: {str(e)}")
            raise

    def _build_chain(self) -> Runnable:
        """
        Build the runnable chain from the configured steps.

        Returns:
            A Runnable representing the complete chain.
        """
        if not self.steps:
            return RunnablePassthrough()

        # Chain all steps together
        full_chain = self.steps[0]
        for step in self.steps[1:]:
            full_chain = full_chain | step

        return full_chain

    def add_step(self, step: Runnable, index: Optional[int] = None) -> None:
        """
        Add a step to the chain.

        Args:
            step: The Runnable step to add.
            index: Optional index to insert at. If None, appends to end.
        """
        if index is not None:
            self.steps = list(self.steps)[:index] + [step] + list(self.steps)[index:]
        else:
            self.steps = list(self.steps) + [step]
        logger.info(f"Added step to sequential chain at index {index}")

    def remove_step(self, index: int) -> None:
        """
        Remove a step from the chain.

        Args:
            index: The index of the step to remove.
        """
        if 0 <= index < len(self.steps):
            self.steps = list(self.steps)[:index] + list(self.steps)[index + 1 :]
            logger.info(f"Removed step at index {index}")

    @classmethod
    def create_research_chain(
        cls,
        llm: BaseLanguageModel,
        include_search: bool = True,
        include_analysis: bool = True,
        include_summary: bool = True,
    ) -> "ResearchSequentialChain":
        """
        Create a pre-configured research chain with common steps.

        Args:
            llm: The language model to use.
            include_search: Whether to include a search step.
            include_analysis: Whether to include an analysis step.
            include_summary: Whether to include a summary step.

        Returns:
            A configured ResearchSequentialChain instance.
        """
        steps = []

        # Search step prompt
        if include_search:
            search_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a research assistant. Given a query, describe what "
                        "information should be searched for to answer it comprehensively.",
                    ),
                    ("human", "Query: {query}"),
                ]
            )
            search_chain = search_prompt | llm
            steps.append(search_chain)

        # Analysis step prompt
        if include_analysis:
            analysis_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert analyst. Analyze the given information and "
                        "identify key findings, patterns, and insights.",
                    ),
                    ("human", "Information: {output}"),
                ]
            )
            analysis_chain = analysis_prompt | llm
            steps.append(analysis_chain)

        # Summary step prompt
        if include_summary:
            summary_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a skilled summarizer. Create a concise summary of "
                        "the key points from the analysis.",
                    ),
                    ("human", "Analysis: {output}"),
                ]
            )
            summary_chain = summary_prompt | llm
            steps.append(summary_chain)

        return cls(steps=steps, llm=llm)
