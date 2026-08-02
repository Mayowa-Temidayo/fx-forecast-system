"""
Project logger.
"""

from __future__ import annotations

import sys

from loguru import logger

from fx_forecast.config.paths import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
)

logger.add(
    LOGS_DIR / "fx_forecast.log",
    level="DEBUG",
    rotation="10 MB",
    retention=5,
    enqueue=True,
)

__all__ = ["logger"]
