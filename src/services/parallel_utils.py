"""Shared utilities for parallel processing across services."""

import logging
from contextlib import contextmanager
from typing import Generator

from tqdm import tqdm


class _TqdmLoggingHandler(logging.Handler):
    """Logging handler that routes messages through tqdm.write() to avoid corrupting progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


_TQDM_LOG_FORMATTER = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@contextmanager
def tqdm_logging() -> Generator[None, None, None]:
    """Context manager that redirects root-logger output through tqdm.write().

    Use this around any block that runs a tqdm progress bar to prevent
    logging output from corrupting the bar display.
    """
    root_logger = logging.getLogger()
    handler = _TqdmLoggingHandler()
    handler.setFormatter(_TQDM_LOG_FORMATTER)
    existing_handlers = root_logger.handlers[:]
    for h in existing_handlers:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        for h in existing_handlers:
            root_logger.addHandler(h)
