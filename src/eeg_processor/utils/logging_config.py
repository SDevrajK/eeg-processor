"""
Centralized logging configuration for EEG Processor.

This module provides consistent logging setup across all modules,
with the option to use either standard logging or loguru.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Try to import loguru, fall back to standard logging if not available
try:
    from loguru import logger as loguru_logger
    HAS_LOGURU = True
except ImportError:
    loguru_logger = None
    HAS_LOGURU = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with consistent configuration.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_loguru: bool = True
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        use_loguru: Whether to use loguru (if available) or standard logging
    """
    if use_loguru and HAS_LOGURU:
        _setup_loguru(level, log_file)
    else:
        _setup_standard_logging(level, log_file)


def _setup_loguru(level: str, log_file: Optional[Path]) -> None:
    """Setup loguru logging configuration."""
    # Remove default handler
    loguru_logger.remove()
    
    # Add console handler with nice formatting
    loguru_logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        loguru_logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="1 week"
        )


def _setup_standard_logging(level: str, log_file: Optional[Path]) -> None:
    """Setup standard library logging configuration."""
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)


# For backward compatibility, provide a logger instance
if HAS_LOGURU:
    logger = loguru_logger
else:
    logger = logging.getLogger(__name__)