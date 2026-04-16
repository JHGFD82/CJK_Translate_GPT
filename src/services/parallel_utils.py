"""Shared utilities for parallel processing across services."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator

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


def update_pbar_postfix(
    pbar: tqdm,
    usage_data: Dict[str, Any],
    baseline_tokens: Any,
    baseline_cost: Any,
) -> None:
    """Update a tqdm progress bar postfix with run-so-far token/cost counts.

    Silently does nothing if the values cannot be converted (e.g. None on
    first call before any usage has been recorded).
    """
    try:
        run_tokens = int(usage_data["total_usage"].get("total_tokens", 0)) - int(baseline_tokens)
        run_cost = float(usage_data["total_usage"].get("total_cost", 0.0)) - float(baseline_cost)
        pbar.set_postfix(tokens=f"{run_tokens:,}", cost=f"${run_cost:.4f}")
    except (TypeError, ValueError):
        pass
