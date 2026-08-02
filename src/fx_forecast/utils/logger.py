"""
Project logger.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
)

logger.add(
    LOG_DIR / "fx_forecast.log",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
)

__all__ = ["logger"]
