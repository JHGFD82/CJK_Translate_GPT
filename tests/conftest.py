"""
Shared pytest fixtures for the PU AI Sandbox test suite.

All services are instantiated with a fake API key and a mocked TokenTracker so
no real network calls or file I/O are triggered during tests.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _no_sleep():
    """Patch time.sleep globally so retry-backoff tests don't actually wait."""
    with patch("time.sleep"):
        yield

from src.services.image_processor_service import ImageProcessorService
from src.services.image_translation_service import ImageTranslationService
from src.services.translation_service import TranslationService


@pytest.fixture
def ocr_service():
    """ImageProcessorService with a fake key and mocked token tracker."""
    return ImageProcessorService(api_key="test-api-key", token_tracker=MagicMock())


@pytest.fixture
def image_translation_service():
    """ImageTranslationService with a fake key and mocked token tracker."""
    return ImageTranslationService(api_key="test-api-key", token_tracker=MagicMock())


@pytest.fixture
def translation_service():
    """TranslationService with a fake key and mocked token tracker."""
    return TranslationService(api_key="test-api-key", token_tracker=MagicMock())
