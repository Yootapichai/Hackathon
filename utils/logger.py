"""
Loguru-based Logging Configuration for Supply Chain Analytics

This module configures loguru for structured logging across all application components.
Replaces the previous standard logging implementation with loguru's simpler and more elegant approach.
"""

import os
import sys
from datetime import datetime
from loguru import logger

def setup_logging(log_level: str = "INFO"):
    """Setup comprehensive logging configuration using loguru"""
    
    # Remove default handler
    logger.remove()
    
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Get current date for log filenames
    current_date = datetime.now().strftime('%Y%m%d')
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        level=log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True
    )
    
    # General application log file
    logger.add(
        f"logs/supply_chain_{current_date}.log",
        level=log_level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="5 days",
        compression="zip"
    )
    
    # Error log file (only errors and critical)
    logger.add(
        f"logs/errors_{current_date}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message} | {exception}",
        rotation="10 MB",
        retention="10 days",
        compression="zip"
    )
    
    # Tool execution log file (debug level for detailed tool operations)
    logger.add(
        f"logs/tools_{current_date}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | TOOL | {level: <8} | {message}",
        rotation="10 MB",
        retention="5 days",
        compression="zip",
        filter=lambda record: "Tool" in record["message"]
    )

def configure_logger():
    """Configure loguru with default settings"""
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(log_level)
    logger.info("Loguru logging configured successfully")

# Auto-configure when module is imported
configure_logger()

# Export logger for easy import
__all__ = ["logger", "setup_logging", "configure_logger"]