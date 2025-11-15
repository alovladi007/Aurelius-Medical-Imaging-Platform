"""Logging utilities for the cancer quantitative model."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    log_dir: Optional[str] = None,
    log_file: str = "experiment.log",
    level: str = "INFO",
    use_rich: bool = True,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_dir: Directory to save log files (if None, only console logging)
        log_file: Name of log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_rich: Use rich formatting for console output

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("cancer_quant_model")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    if use_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_format)

    logger.addHandler(console_handler)

    # File handler (if log_dir specified)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / log_file)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "cancer_quant_model") -> logging.Logger:
    """Get logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporary logging level changes."""

    def __init__(self, logger: logging.Logger, level: str):
        """Initialize logger context.

        Args:
            logger: Logger instance
            level: Temporary logging level
        """
        self.logger = logger
        self.level = getattr(logging, level.upper())
        self.original_level = logger.level

    def __enter__(self):
        """Enter context."""
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.logger.setLevel(self.original_level)


def log_params(logger: logging.Logger, params: dict, title: str = "Parameters"):
    """Log parameters in a formatted way.

    Args:
        logger: Logger instance
        params: Parameters dictionary
        title: Title for the log section
    """
    logger.info(f"\n{'=' * 50}")
    logger.info(f"{title}")
    logger.info(f"{'=' * 50}")

    for key, value in params.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")

    logger.info(f"{'=' * 50}\n")
