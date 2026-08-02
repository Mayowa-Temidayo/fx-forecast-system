"""
Central project logger.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

logger.add(
    sys.stdout,
    level="INFO",
    format=LOG_FORMAT,
    colorize=True,
)

logger.add(
    LOG_DIR / "fx_forecast.log",
    level="DEBUG",
    format=LOG_FORMAT,
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    encoding="utf-8",
)

__all__ = ["logger"]
