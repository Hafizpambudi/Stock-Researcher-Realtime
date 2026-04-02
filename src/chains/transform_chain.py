"""
Transform Chain implementation for the Research Assistant.

This module implements a transform chain that applies transformations
to input data, such as formatting, filtering, or enriching content.
"""

from typing import Any, Callable, Optional


from langchain_core.callbacks.manager import Callbacks
from langchain_classic.chains.base import Chain
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import Field, validator

from src.utils.helpers import sanitize_text, truncate_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransformConfig:
    """Configuration for transform operations."""

    @staticmethod
    def uppercase(text: str) -> str:
        """Transform text to uppercase."""
        return text.upper()

    @staticmethod
    def lowercase(text: str) -> str:
        """Transform text to lowercase."""
        return text.lower()

    @staticmethod
    def title_case(text: str) -> str:
        """Transform text to title case."""
        return text.title()

    @staticmethod
    def sanitize(text: str) -> str:
        """Sanitize text by removing excessive whitespace."""
        return sanitize_text(text)

    @staticmethod
    def truncate(text: str, max_length: int = 500) -> str:
        """Truncate text to maximum length."""
        return truncate_text(text, max_length)

    @staticmethod
    def extract_key_points(text: str) -> str:
        """Extract key points as bullet list."""
        sentences = text.replace("\n", " ").split(". ")
        key_points = [s.strip() + "." for s in sentences[:5] if len(s.strip()) > 10]
        return "\n".join(f"• {pt}" for pt in key_points)

    @staticmethod
    def format_as_markdown(text: str, title: str = "Report") -> str:
        """Format text as markdown with heading."""
        return f"# {title}\n\n{text}"


class ResearchTransformChain(Chain):
    """
    A transform chain for applying transformations to research data.

    This chain applies one or more transformations to input data,
    such as formatting, filtering, enriching, or restructuring content.
    It's useful for preparing research output in different formats.

    Attributes:
        name: The name of the chain.
        transforms: A list of transformation functions to apply.
        llm: Optional language model for LLM-based transformations.
        input_key: The key for the chain's input.
        output_key: The key for the chain's output.

    Example:
        >>> chain = ResearchTransformChain(
        ...     transforms=[
        ...         TransformConfig.sanitize,
        ...         TransformConfig.extract_key_points,
        ...     ]
        ... )
        >>> result = chain.invoke({"content": "Raw text..."})
    """

    name: str = "research_transform_chain"
    input_key: str = "content"
    output_key: str = "transformed"

    transforms: list[Callable[[str], str]] = Field(default_factory=list)
    llm: Optional[BaseLanguageModel] = Field(default=None)
    transform_config: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @validator("transforms", pre=True)
    def validate_transforms(cls, v: Any) -> Any:
        """Validate that transforms are callable."""
        if v is None:
            return []
        return v

    def __init__(
        self,
        transforms: Optional[list[Callable[[str], str]]] = None,
        llm: Optional[BaseLanguageModel] = None,
        transform_config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Transform Chain.

        Args:
            transforms: Optional list of transformation functions.
            llm: Optional language model for LLM-based transformations.
            transform_config: Optional configuration for transformations.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            transforms=transforms or [],
            llm=llm,
            transform_config=transform_config or {},
            **kwargs,
        )

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
        Execute the transform chain with the given inputs.

        Args:
            inputs: Dictionary containing the input values.
            run_manager: Optional callback manager for execution tracking.

        Returns:
            Dictionary containing the transformed output.
        """
        content = inputs.get(self.input_key, "")
        logger.info(f"Executing transform chain with {len(self.transforms)} transforms")

        try:
            result = content

            # Apply each transform in sequence
            for transform in self.transforms:
                result = transform(result)

            # Apply LLM-based transformation if configured
            if self.llm and self.transform_config.get("llm_transform"):
                result = self._apply_llm_transform(result)

            logger.info("Transform chain execution completed")
            return {self.output_key: result}

        except Exception as e:
            logger.error(f"Transform chain execution failed: {str(e)}")
            raise

    def _apply_llm_transform(self, content: str) -> str:
        """
        Apply an LLM-based transformation to the content.

        Args:
            content: The content to transform.

        Returns:
            The transformed content.
        """
        transform_type = self.transform_config.get("llm_transform", "summarize")

        prompts = {
            "summarize": ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Summarize the following content concisely while preserving key information.",
                    ),
                    ("human", "{content}"),
                ]
            ),
            "expand": ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Expand on the following content with additional relevant details and context.",
                    ),
                    ("human", "{content}"),
                ]
            ),
            "rephrase": ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Rephrase the following content in a clearer, more professional tone.",
                    ),
                    ("human", "{content}"),
                ]
            ),
            "translate": ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Translate the following content to {target_language}.",
                    ),
                    ("human", "{content}"),
                ]
            ),
        }

        prompt_template = prompts.get(transform_type, prompts["summarize"])

        # Add target language for translation
        format_args = {"content": content}
        if transform_type == "translate":
            format_args["target_language"] = self.transform_config.get(
                "target_language", "Spanish"
            )

        prompt = prompt_template.format(**format_args)
        response = self.llm.invoke(prompt)

        return response.content if hasattr(response, "content") else str(response)

    def add_transform(self, transform: Callable[[str], str]) -> None:
        """
        Add a transformation function to the chain.

        Args:
            transform: The transformation function to add.
        """
        self.transforms.append(transform)
        logger.info(f"Added transform: {transform.__name__}")

    def clear_transforms(self) -> None:
        """Clear all transformation functions."""
        self.transforms.clear()
        logger.info("Cleared all transforms")

    @classmethod
    def create_format_chain(
        cls,
        output_format: str = "markdown",
        llm: Optional[BaseLanguageModel] = None,
    ) -> "ResearchTransformChain":
        """
        Create a pre-configured chain for formatting output.

        Args:
            output_format: The desired output format (markdown, html, plain).
            llm: Optional language model for enhanced formatting.

        Returns:
            A configured ResearchTransformChain instance.
        """
        transforms = [TransformConfig.sanitize]

        if output_format == "markdown":
            transforms.append(
                lambda x: TransformConfig.format_as_markdown(x, "Research Report")
            )
        elif output_format == "bullet_points":
            transforms.append(TransformConfig.extract_key_points)

        return cls(transforms=transforms, llm=llm)

    @classmethod
    def create_summary_chain(
        cls,
        llm: BaseLanguageModel,
        max_length: int = 500,
    ) -> "ResearchTransformChain":
        """
        Create a pre-configured chain for summarization.

        Args:
            llm: The language model to use.
            max_length: Maximum length of the summary.

        Returns:
            A configured ResearchTransformChain instance.
        """
        chain = cls(
            transforms=[TransformConfig.sanitize],
            llm=llm,
            transform_config={"llm_transform": "summarize"},
        )
        return chain

    @classmethod
    def create_enrichment_chain(
        cls,
        llm: BaseLanguageModel,
    ) -> "ResearchTransformChain":
        """
        Create a pre-configured chain for content enrichment.

        Args:
            llm: The language model to use.

        Returns:
            A configured ResearchTransformChain instance.
        """
        chain = cls(
            transforms=[TransformConfig.sanitize],
            llm=llm,
            transform_config={"llm_transform": "expand"},
        )
        return chain
