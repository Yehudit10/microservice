import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)
    if log_file:
        handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(handler)
