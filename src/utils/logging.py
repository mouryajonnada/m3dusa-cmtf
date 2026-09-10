"""
Logging Setup
=============
Configures a Python ``logging.Logger`` that writes to both stdout and a
rotating log file under ``cfg.results.output_dir/logs/``.

Usage::

    from src.utils.logging import get_logger

    log = get_logger(__name__, cfg)
    log.info("Training started")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger writing to stdout and optionally a file.

    Args:
        name:    Logger name (use ``__name__`` in callers).
        log_dir: Directory to write ``<name>.log`` into. If *None*, file
                 logging is disabled.
        level:   Logging level (default: ``logging.INFO``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (e.g., re-imported)

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name.replace('.', '_')}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
