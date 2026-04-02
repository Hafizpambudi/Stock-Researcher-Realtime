"""
Helper utilities for the Research Assistant.

This module provides common utility functions used throughout
the application for data processing, formatting, and validation.
"""

import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from tenacity import retry, stop_after_attempt, wait_exponential


def generate_id() -> str:
    """
    Generate a unique identifier.

    Returns:
        A unique string identifier (UUID4 format).

    Example:
        >>> id = generate_id()
        >>> print(id)
        '550e8400-e29b-41d4-a716-446655440000'
    """
    return str(uuid.uuid4())


def generate_hash(content: str, algorithm: str = "sha256") -> str:
    """
    Generate a hash of the given content.

    Args:
        content: The string content to hash.
        algorithm: The hashing algorithm to use (md5, sha1, sha256).
            Defaults to "sha256".

    Returns:
        The hexadecimal hash string.

    Example:
        >>> hash_value = generate_hash("Hello, World!")
    """
    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    return hash_func(content.encode()).hexdigest()


def sanitize_text(text: str) -> str:
    """
    Sanitize text by removing excessive whitespace and normalizing.

    Args:
        text: The text to sanitize.

    Returns:
        The sanitized text with normalized whitespace.

    Example:
        >>> sanitize_text("Hello   World!\\n\\n")
        'Hello World!'
    """
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: The text to truncate.
        max_length: Maximum length of the output text. Defaults to 1000.
        suffix: Suffix to add if text is truncated. Defaults to "...".

    Returns:
        The truncated text with suffix if applicable.

    Example:
        >>> truncate_text("A very long text...", max_length=20)
        'A very long te...'
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_timestamp(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime object as a string.

    Args:
        dt: The datetime to format. If None, uses current time.
        format_str: The format string. Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        The formatted timestamp string.

    Example:
        >>> format_timestamp()
        '2024-01-15 10:30:00'
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format_str)


def parse_markdown_links(text: str) -> list[dict[str, str]]:
    """
    Parse markdown links from text.

    Args:
        text: The text containing markdown links.

    Returns:
        A list of dictionaries with 'text' and 'url' keys.

    Example:
        >>> parse_markdown_links("Check [Google](https://google.com)")
        [{'text': 'Google', 'url': 'https://google.com'}]
    """
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    matches = re.findall(pattern, text)
    return [{"text": match[0], "url": match[1]} for match in matches]


def extract_urls(text: str) -> list[str]:
    """
    Extract URLs from text.

    Args:
        text: The text containing URLs.

    Returns:
        A list of extracted URLs.

    Example:
        >>> extract_urls("Visit https://example.com for more info")
        ['https://example.com']
    """
    pattern = r"https?://[^\s<>\[\]\"']+"
    return re.findall(pattern, text)


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback to default.

    Args:
        json_str: The JSON string to parse.
        default: Default value if parsing fails. Defaults to None.

    Returns:
        The parsed JSON object or the default value.

    Example:
        >>> safe_json_loads('{"key": "value"}')
        {'key': 'value'}
        >>> safe_json_loads('invalid json', default={})
        {}
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """
    Safely serialize object to JSON string.

    Args:
        obj: The object to serialize.
        **kwargs: Additional arguments passed to json.dumps.

    Returns:
        The JSON string representation.

    Example:
        >>> safe_json_dumps({"key": "value"})
        '{"key": "value"}'
    """
    try:
        return json.dumps(obj, **kwargs)
    except (TypeError, ValueError):
        return "{}"


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: The path to the directory.

    Returns:
        The Path object for the directory.

    Example:
        >>> ensure_directory("./data/output")
        PosixPath('data/output')
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts. Defaults to 3.
        initial_delay: Initial delay in seconds. Defaults to 1.0.
        max_delay: Maximum delay in seconds. Defaults to 60.0.
        exponential_base: Base for exponential backoff. Defaults to 2.0.

    Returns:
        A decorator that adds retry logic to the function.

    Example:
        >>> @retry_with_backoff(max_attempts=5)
        ... def api_call():
        ...     # API call that might fail
        ...     pass
    """

    def decorator(func):
        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=initial_delay,
                exp_base=exponential_base,
                max=max_delay,
            ),
            reraise=True,
        )(func)

    return decorator


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk.
        chunk_size: Maximum size of each chunk. Defaults to 1000.
        overlap: Number of characters to overlap between chunks. Defaults to 100.

    Returns:
        A list of text chunks.

    Example:
        >>> chunks = chunk_text("A long text...", chunk_size=100, overlap=20)
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)

            if break_point > chunk_size // 2:
                chunk = chunk[: break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: The base dictionary.
        override: The dictionary with values to override.

    Returns:
        A new dictionary with merged values.

    Example:
        >>> merge_dicts({"a": 1, "b": {"c": 2}}, {"b": {"d": 3}})
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result
