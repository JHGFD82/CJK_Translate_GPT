"""Shared API error classification utilities for all service modules."""

import logging

from ..models import is_model_access_error, remove_model_from_catalog


def is_content_filter_error(error: Exception) -> bool:
    """Return True if the error was caused by a content filter or jailbreak response."""
    msg = str(error).lower()
    return "content_filter" in msg or "jailbreak" in msg


def raise_for_model_access_error(error: Exception, model: str) -> None:
    """Raise a user-friendly ValueError if *error* is a model-access denial.

    Removes the model from the catalog before raising so subsequent calls
    will not attempt to use it again. Does nothing if the error is not a
    model-access denial.
    """
    if not is_model_access_error(str(error)):
        return
    removed = remove_model_from_catalog(model) if model else False
    removed_note = " It has been removed from the catalog." if removed else ""
    logging.error(f"Model access denied for {model!r}: {error}")
    raise ValueError(
        f"Model '{model}' is not accessible in the Princeton AI Sandbox — "
        f"you do not have access to this model.{removed_note} "
        "Please use a different model or contact your sandbox administrator."
    ) from error


def handle_common_api_errors(error: Exception, model: str) -> None:
    """Raise a user-friendly exception for common PortKey/OpenAI API errors.

    Covers model-access denial, rate limits, invalid requests, and
    authentication failures. Content-filter and context-length errors are
    intentionally excluded — callers that need signal-based handling (e.g.
    TranslationService) deal with those themselves.

    If none of the known patterns match, this function returns without raising
    so the caller can decide how to handle the remaining error.
    """
    msg = str(error).lower()
    raise_for_model_access_error(error, model)
    if "rate_limit" in msg:
        logging.error(f"Rate limit exceeded: {error}")
        raise Exception(f"Rate limit exceeded: {error}") from error
    if "invalid_request" in msg:
        logging.error(f"Invalid request: {error}")
        raise Exception(f"Invalid request: {error}") from error
    if "authentication" in msg or "unauthorized" in msg:
        logging.error(f"Authentication error: {error}")
        raise Exception(f"Authentication error: {error}") from error
