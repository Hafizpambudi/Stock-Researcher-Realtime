"""
Logging configuration for the Research Assistant.

This module provides centralized logging setup with support for
console and file output, customizable log levels, and structured logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """
    Configure the application logger with console and optional file output.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to "INFO".
        log_file: Optional path to a log file. If None, only console output
            is configured. Defaults to None.
        log_format: Optional custom log format string. If None, uses the
            default format. Defaults to None.

    Example:
        >>> setup_logger(log_level="DEBUG", log_file="./logs/app.log")
    """
    # Remove default handler
    logger.remove()

    # Default format
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            format=log_format,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

    logger.info(f"Logger initialized with level: {log_level}")


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger instance for the specified module name.

    This function provides compatibility with the standard logging module
    while using loguru under the hood.

    Args:
        name: The name of the logger, typically __name__ of the calling module.

    Returns:
        A logger instance configured for the specified name.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process")
    """
    # Intercept standard logging and redirect to loguru
    logger.configure(handlers=[{"sink": sys.stderr, "format": "{message}"}])

    class InterceptHandler(logging.Handler):
        """Intercept standard logging and redirect to loguru."""

        def emit(self, record: logging.LogRecord) -> None:
            """Emit a log record to loguru."""
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    return logging.getLogger(name)


# Default logger instance for module-level use
default_logger = get_logger(__name__)
