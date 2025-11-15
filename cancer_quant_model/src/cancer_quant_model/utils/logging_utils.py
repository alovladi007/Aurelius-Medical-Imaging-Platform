"""Logging utilities for the cancer quantitative model."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "cancer_quant_model",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_rich: Use Rich for beautiful console output

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    if use_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=True,
            show_path=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "cancer_quant_model") -> logging.Logger:
    """
    Get an existing logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporary logger level changes."""

    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize logger context.

        Args:
            logger: Logger to modify
            level: Temporary logging level
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = None

    def __enter__(self):
        """Enter context."""
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.logger.setLevel(self.old_level)


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = ""):
    """
    Log metrics in a formatted way.

    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        prefix: Optional prefix for metric names
    """
    if prefix:
        prefix = f"{prefix}/"

    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{prefix}{key}: {value:.4f}")
        else:
            logger.info(f"{prefix}{key}: {value}")
