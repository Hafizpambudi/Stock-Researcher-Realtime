"""
Tests for utility modules.

This module contains unit tests for the logger, config, and helpers modules.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Settings, get_env_variable, get_settings, load_config, validate_environment
from src.utils.helpers import (
    chunk_text,
    extract_urls,
    format_timestamp,
    generate_hash,
    generate_id,
    merge_dicts,
    parse_markdown_links,
    safe_json_dumps,
    safe_json_loads,
    sanitize_text,
    truncate_text,
)


class TestHelpers:
    """Test cases for helper functions."""

    def test_generate_id(self) -> None:
        """Test ID generation produces unique UUIDs."""
        id1 = generate_id()
        id2 = generate_id()
        assert id1 != id2
        assert len(id1) == 36  # UUID4 format

    def test_generate_hash(self) -> None:
        """Test hash generation."""
        content = "Hello, World!"
        hash1 = generate_hash(content)
        hash2 = generate_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

        # Different algorithms
        md5_hash = generate_hash(content, algorithm="md5")
        assert len(md5_hash) == 32

    def test_sanitize_text(self) -> None:
        """Test text sanitization."""
        assert sanitize_text("Hello   World") == "Hello World"
        assert sanitize_text("  Trimmed  ") == "Trimmed"
        assert sanitize_text("Multiple\n\nnewlines") == "Multiple newlines"

    def test_truncate_text(self) -> None:
        """Test text truncation."""
        long_text = "A" * 100
        assert len(truncate_text(long_text, max_length=50)) <= 53  # 50 + "..."
        assert truncate_text("Short", max_length=100) == "Short"
        assert truncate_text(long_text, max_length=10, suffix="..") == "A" * 8 + ".."

    def test_format_timestamp(self) -> None:
        """Test timestamp formatting."""
        timestamp = format_timestamp()
        assert len(timestamp) > 0
        assert "-" in timestamp  # Date format

    def test_parse_markdown_links(self) -> None:
        """Test markdown link parsing."""
        text = "Check [Google](https://google.com) and [GitHub](https://github.com)"
        links = parse_markdown_links(text)
        assert len(links) == 2
        assert links[0]["text"] == "Google"
        assert links[0]["url"] == "https://google.com"

    def test_extract_urls(self) -> None:
        """Test URL extraction."""
        text = "Visit https://example.com or http://test.org/page"
        urls = extract_urls(text)
        assert len(urls) == 2
        assert "https://example.com" in urls

    def test_safe_json_loads(self) -> None:
        """Test safe JSON parsing."""
        assert safe_json_loads('{"key": "value"}') == {"key": "value"}
        assert safe_json_loads("invalid", default={}) == {}
        assert safe_json_loads(None, default="default") == "default"

    def test_safe_json_dumps(self) -> None:
        """Test safe JSON serialization."""
        assert safe_json_dumps({"key": "value"}) == '{"key": "value"}'
        assert safe_json_dumps(set([1, 2, 3]), default="{}") == "{}"

    def test_chunk_text(self) -> None:
        """Test text chunking."""
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1
        assert all(len(chunk) <= 350 for chunk in chunks)  # Allow some overlap

        # Short text should return as single chunk
        assert chunk_text("Short", chunk_size=100) == ["Short"]

    def test_merge_dicts(self) -> None:
        """Test dictionary merging."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10, "e": 5}, "f": 6}
        result = merge_dicts(base, override)

        assert result["a"] == 1
        assert result["b"]["c"] == 10  # Overridden
        assert result["b"]["d"] == 3  # Preserved
        assert result["b"]["e"] == 5  # Added
        assert result["f"] == 6  # Added


class TestConfig:
    """Test cases for configuration functions."""

    def test_get_env_variable(self) -> None:
        """Test environment variable retrieval."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            assert get_env_variable("TEST_VAR") == "test_value"
            assert get_env_variable("NONEXISTENT", "default") == "default"
            assert get_env_variable("NONEXISTENT") is None

    def test_validate_environment(self) -> None:
        """Test environment validation."""
        with patch.dict(os.environ, {}, clear=True):
            is_valid, missing = validate_environment()
            assert is_valid is False
            assert "OPENAI_API_KEY" in missing

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            is_valid, missing = validate_environment()
            assert is_valid is True
            assert len(missing) == 0

    def test_get_settings_default(self) -> None:
        """Test settings with defaults."""
        # Clear environment to test defaults
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.openai_model == "gpt-4-turbo-preview"
            assert settings.max_search_results == 5
            assert settings.log_level == "INFO"

    def test_load_config_structure(self) -> None:
        """Test config loading returns expected structure."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config()
            assert "openai" in config
            assert "search" in config
            assert "logging" in config
            assert config["search"]["max_results"] == 5


class TestLogger:
    """Test cases for logging configuration."""

    def test_logger_setup(self) -> None:
        """Test logger setup doesn't raise errors."""
        from src.utils.logger import setup_logger

        # Should not raise
        setup_logger(log_level="DEBUG")
        setup_logger(log_level="INFO", log_file=None)

    def test_get_logger(self) -> None:
        """Test getting a logger instance."""
        from src.utils.logger import get_logger

        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
