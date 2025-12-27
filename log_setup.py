import logging
import sys
from pathlib import Path

import os

# Determine if running in Docker (by checking for /.dockerenv or DOCKER env var)
def is_docker():
    return os.path.exists('/.dockerenv') or os.environ.get('DOCKER', '').lower() == 'true'

if is_docker():
    log_path = os.environ.get("LOG_FILE", "/app/logs/app.log")
else:
    log_path = os.environ.get("LOG_FILE", "logs/app.log")

LOG_FILE = Path(log_path)

# Ensure the parent directory exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


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
