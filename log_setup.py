import logging
import sys
from pathlib import Path

import os

# Use /app/logs/app.log inside the container for persistent logging
LOG_FILE = Path(os.environ.get("LOG_FILE", "/app/logs/app.log"))


def configure_logging(level=logging.INFO):
    """Configure root logger to output to stdout and a rotating file.

    Returns the root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)

    # File handler (append)
    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # Avoid adding multiple handlers if already configured
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
