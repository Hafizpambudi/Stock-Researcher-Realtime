"""
Configuration management for the Research Assistant.

This module handles loading configuration from environment variables
and configuration files, providing type-safe access to settings.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    This class defines all configurable parameters for the Research Assistant,
    with sensible defaults and type validation via Pydantic.

    Attributes:
        openrouter_api_key: API key for OpenRouter services.
        openrouter_base_url: Base URL for OpenRouter API.
        openrouter_model: The model to use via OpenRouter.
        openrouter_reasoning_enabled: Whether to enable OpenRouter reasoning feature.
        openai_api_key: API key for OpenAI services (direct).
        openai_model: The OpenAI model to use for completions (direct).
        anthropic_api_key: API key for Anthropic services.
        anthropic_model: The Anthropic model to use for completions.
        search_engine: The search engine to use (duckduckgo, google, etc.).
        max_search_results: Maximum number of search results to return.
        vector_store_path: Path to store vector embeddings.
        embedding_model: The embedding model to use for vectorization.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file.
        max_retries: Maximum number of retry attempts for API calls.
        retry_delay: Delay between retries in seconds.
        default_report_format: Default format for generated reports.
        include_citations: Whether to include citations in reports.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
    )

    # OpenRouter Configuration (Primary)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = ""  # Will be loaded from OpenRouterModel env var
    openrouter_reasoning_enabled: bool = False

    # OpenAI Configuration (Direct - Optional)
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-preview"

    # Anthropic Configuration
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-opus-20240229"

    # Search Configuration
    search_engine: str = "duckduckgo"
    max_search_results: int = 5

    # Vector Store Configuration
    vector_store_path: str = "./data/vector_store"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "./logs/research_assistant.log"

    # Rate Limiting
    max_retries: int = 3
    retry_delay: float = 1.0

    # Report Configuration
    default_report_format: str = "markdown"
    include_citations: bool = True


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    This function uses LRU caching to avoid reloading settings on every call.
    The settings are loaded from environment variables and .env file.

    Returns:
        Settings: The application settings instance.

    Example:
        >>> settings = get_settings()
        >>> print(settings.openrouter_model)
    """
    # Load .env file from project root
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Try loading from current directory
        load_dotenv()

    settings = Settings()

    # Override openrouter_model with OpenRouterModel env var if set
    # This allows users to specify the model via OpenRouterModel variable
    openrouter_model_env = os.environ.get("OpenRouterModel")
    if openrouter_model_env:
        settings.openrouter_model = openrouter_model_env
    elif not settings.openrouter_model:
        # Fallback to default if neither is set
        settings.openrouter_model = "openai/gpt-4-turbo"

    # Override openrouter_reasoning_enabled with environment variable if set
    reasoning_env = os.environ.get("OPENROUTER_REASONING_ENABLED", "").lower()
    if reasoning_env in ("true", "1", "yes"):
        settings.openrouter_reasoning_enabled = True
    elif reasoning_env in ("false", "0", "no"):
        settings.openrouter_reasoning_enabled = False

    return settings


def load_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """
    Load configuration from a file or environment variables.

    Args:
        config_path: Optional path to a configuration file. If None,
            loads from environment variables only.

    Returns:
        A dictionary containing the configuration values.

    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist.

    Example:
        >>> config = load_config("./config/settings.yaml")
    """
    settings = get_settings()

    config = {
        "openrouter": {
            "api_key": settings.openrouter_api_key,
            "base_url": settings.openrouter_base_url,
            "model": settings.openrouter_model,
        },
        "openai": {
            "api_key": settings.openai_api_key,
            "model": settings.openai_model,
        },
        "anthropic": {
            "api_key": settings.anthropic_api_key,
            "model": settings.anthropic_model,
        },
        "search": {
            "engine": settings.search_engine,
            "max_results": settings.max_search_results,
        },
        "vector_store": {
            "path": settings.vector_store_path,
            "embedding_model": settings.embedding_model,
        },
        "logging": {
            "level": settings.log_level,
            "file": settings.log_file,
        },
        "retry": {
            "max_attempts": settings.max_retries,
            "delay": settings.retry_delay,
        },
        "report": {
            "format": settings.default_report_format,
            "include_citations": settings.include_citations,
        },
    }

    # Load additional config from file if specified
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            # Support YAML configuration
            if config_file.suffix in [".yaml", ".yml"]:
                try:
                    import yaml

                    with open(config_file, "r") as f:
                        file_config = yaml.safe_load(f)
                        config.update(file_config)
                except ImportError:
                    pass  # yaml not installed, skip file config

    return config


def get_env_variable(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable value.

    Args:
        name: The name of the environment variable.
        default: Optional default value if the variable is not set.

    Returns:
        The value of the environment variable, or the default if not set.

    Example:
        >>> api_key = get_env_variable("OPENROUTER_API_KEY")
    """
    return os.environ.get(name, default)


def validate_environment() -> tuple[bool, list[str]]:
    """
    Validate that required environment variables are set.

    Returns:
        A tuple of (is_valid, missing_variables) where:
        - is_valid: True if all required variables are set
        - missing_variables: List of missing variable names

    Example:
        >>> is_valid, missing = validate_environment()
        >>> if not is_valid:
        ...     print(f"Missing: {missing}")
    """
    # Check for OpenRouter API key (primary) or OpenAI API key (fallback)
    required_vars = []
    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        required_vars.append("OPENROUTER_API_KEY or OPENAI_API_KEY")

    return len(required_vars) == 0, required_vars


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.7,
    use_openrouter: bool = True,
    reasoning_enabled: bool = False,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    Get a ChatOpenAI instance configured for OpenRouter or direct OpenAI.

    This function creates a ChatOpenAI LLM instance with the appropriate
    configuration for OpenRouter API (default) or direct OpenAI API.

    Args:
        model: Optional model name. If not provided, uses the configured
            default from environment variables.
        temperature: The temperature for the LLM (default: 0.7).
        use_openrouter: If True (default), configure for OpenRouter API.
            If False, use direct OpenAI API.
        reasoning_enabled: Whether to enable OpenRouter reasoning feature (default: False).
            When enabled, the model will provide reasoning traces for its responses.
            Only supported by specific models via OpenRouter.
        **kwargs: Additional keyword arguments passed to ChatOpenAI.

    Returns:
        A configured ChatOpenAI instance.

    Example:
        >>> # Using OpenRouter (default)
        >>> llm = get_llm()
        >>>
        >>> # Using specific model via OpenRouter
        >>> llm = get_llm(model="anthropic/claude-3-opus")
        >>>
        >>> # Using OpenRouter with reasoning enabled
        >>> llm = get_llm(model="minimax/minimax-m2.5", reasoning_enabled=True)
        >>>
        >>> # Using direct OpenAI
        >>> llm = get_llm(use_openrouter=False)
    """
    settings = get_settings()

    if use_openrouter:
        # Configure for OpenRouter API
        api_key = kwargs.pop("api_key", None) or settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        base_url = kwargs.pop("base_url", None) or settings.openrouter_base_url
        model_name = os.environ.get("OPENROUTER_MODEL", "")

        # OpenRouter requires a special header for ranking
        extra_headers = kwargs.pop("extra_headers", {})
        extra_headers["HTTP-Referer"] = os.environ.get("OPENROUTER_REFERER", "https://localhost")
        extra_headers["X-Title"] = os.environ.get("OPENROUTER_TITLE", "Research Assistant")

        # Build model_kwargs for OpenRouter-specific features
        model_kwargs = kwargs.pop("model_kwargs", {})
        
        # Add reasoning configuration if enabled
        if reasoning_enabled or settings.openrouter_reasoning_enabled:
            model_kwargs["reasoning"] = {"enabled": True}

        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=temperature,
            extra_headers=extra_headers,
            model_kwargs=model_kwargs,
            **kwargs,
        )
    else:
        # Configure for direct OpenAI API
        api_key = kwargs.pop("api_key", None) or settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        model_name = model or settings.openai_model

        return ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            **kwargs,
        )


def get_openrouter_llm(
    model: Optional[str] = None,
    temperature: float = 0.7,
    reasoning_enabled: bool = False,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    Get a ChatOpenAI instance specifically configured for OpenRouter API.

    This is a convenience wrapper around get_llm() that always uses OpenRouter.

    Args:
        model: Optional model name. If not provided, uses OPENROUTER_MODEL
            from environment variables.
        temperature: The temperature for the LLM (default: 0.7).
        reasoning_enabled: Whether to enable OpenRouter reasoning feature (default: False).
            When enabled, the model will provide reasoning traces for its responses.
            Only supported by specific models (e.g., minimax/minimax-m2.5, some Claude models).
        **kwargs: Additional keyword arguments passed to ChatOpenAI.

    Returns:
        A configured ChatOpenAI instance for OpenRouter.

    Example:
        >>> llm = get_openrouter_llm()
        >>> llm = get_openrouter_llm(model="google/gemini-pro")
        >>> llm = get_openrouter_llm(model="minimax/minimax-m2.5", reasoning_enabled=True)
    """
    return get_llm(
        model=model,
        temperature=temperature,
        use_openrouter=True,
        reasoning_enabled=reasoning_enabled,
        **kwargs,
    )


def get_openai_llm(
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    Get a ChatOpenAI instance configured for direct OpenAI API.

    This is a convenience wrapper for direct OpenAI usage.

    Args:
        model: Optional model name. If not provided, uses OPENAI_MODEL
            from environment variables.
        temperature: The temperature for the LLM (default: 0.7).
        **kwargs: Additional keyword arguments passed to ChatOpenAI.

    Returns:
        A configured ChatOpenAI instance for direct OpenAI.

    Example:
        >>> llm = get_openai_llm()
        >>> llm = get_openai_llm(model="gpt-4-turbo-preview")
    """
    return get_llm(model=model, temperature=temperature, use_openrouter=False, **kwargs)
