"""
Tests for BaseService sampling-parameter override logic.

No API calls are made; _create_completion is patched so we can inspect
the exact kwargs forwarded to the Portkey client.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.services.translation_service import TranslationService, TRANSLATION_TEMPERATURE, TRANSLATION_TOP_P
from src.services.image_processor_service import ImageProcessorService, OCR_TEMPERATURE, OCR_TOP_P
from src.services.prompt_service import PromptService, PROMPT_TEMPERATURE, PROMPT_TOP_P
from src.services.image_translation_service import ImageTranslationService, IMAGE_TRANSLATION_TEMPERATURE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def translation_svc():
    return TranslationService(api_key="fake", token_tracker=MagicMock())

@pytest.fixture
def translation_svc_custom():
    return TranslationService(api_key="fake", token_tracker=MagicMock(), temperature=0.1, top_p=0.2)

@pytest.fixture
def ocr_svc():
    return ImageProcessorService(api_key="fake", token_tracker=MagicMock())

@pytest.fixture
def ocr_svc_custom():
    return ImageProcessorService(api_key="fake", token_tracker=MagicMock(), temperature=0.5, top_p=0.6)

@pytest.fixture
def prompt_svc():
    return PromptService(api_key="fake", token_tracker=MagicMock())

@pytest.fixture
def prompt_svc_custom():
    return PromptService(api_key="fake", token_tracker=MagicMock(), temperature=0.3, top_p=0.4)

@pytest.fixture
def img_trans_svc():
    return ImageTranslationService(api_key="fake", token_tracker=MagicMock())

@pytest.fixture
def img_trans_svc_custom():
    return ImageTranslationService(api_key="fake", token_tracker=MagicMock(), temperature=0.1)


# ---------------------------------------------------------------------------
# BaseService attribute initialisation
# ---------------------------------------------------------------------------

class TestBaseServiceInit:

    def test_no_overrides_stored_as_none(self, translation_svc):
        assert translation_svc.custom_temperature is None
        assert translation_svc.custom_top_p is None

    def test_custom_temperature_stored(self, translation_svc_custom):
        assert translation_svc_custom.custom_temperature == pytest.approx(0.1)

    def test_custom_top_p_stored(self, translation_svc_custom):
        assert translation_svc_custom.custom_top_p == pytest.approx(0.2)

    def test_ocr_custom_temperature_stored(self, ocr_svc_custom):
        assert ocr_svc_custom.custom_temperature == pytest.approx(0.5)

    def test_prompt_custom_top_p_stored(self, prompt_svc_custom):
        assert prompt_svc_custom.custom_top_p == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# TranslationService — override vs. default
# ---------------------------------------------------------------------------

class TestTranslationServiceSamplingOverride:

    def test_default_temperature_used_when_no_override(self, translation_svc):
        captured = {}

        def fake_create(model, messages, max_tokens, temperature=None, top_p=None, **kw):
            captured["temperature"] = temperature
            captured["top_p"] = top_p

        with patch.object(translation_svc, "_create_completion", side_effect=fake_create):
            translation_svc._call_translation_api("gpt-4o", "system", "sys prompt", "user prompt")

        assert captured["temperature"] == pytest.approx(TRANSLATION_TEMPERATURE)
        assert captured["top_p"] == pytest.approx(TRANSLATION_TOP_P)

    def test_custom_temperature_overrides_default(self, translation_svc_custom):
        captured = {}

        def fake_create(model, messages, max_tokens, temperature=None, top_p=None, **kw):
            captured["temperature"] = temperature
            captured["top_p"] = top_p

        with patch.object(translation_svc_custom, "_create_completion", side_effect=fake_create):
            translation_svc_custom._call_translation_api("gpt-4o", "system", "sys prompt", "user prompt")

        assert captured["temperature"] == pytest.approx(0.1)
        assert captured["top_p"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# ImageProcessorService — override vs. default
# ---------------------------------------------------------------------------

class TestOCRServiceSamplingOverride:

    def test_default_temperature_used_when_no_override(self, ocr_svc):
        captured = {}

        def fake_create(model, messages, max_tokens, temperature=None, top_p=None, **kw):
            captured["temperature"] = temperature
            captured["top_p"] = top_p

        with patch.object(ocr_svc, "_create_completion", side_effect=fake_create):
            ocr_svc._call_ocr_api("gpt-4o", "system", "sys prompt", "user prompt", "data:image/png;base64,abc", 100)

        assert captured["temperature"] == pytest.approx(OCR_TEMPERATURE)
        assert captured["top_p"] == pytest.approx(OCR_TOP_P)

    def test_custom_values_override_ocr_defaults(self, ocr_svc_custom):
        captured = {}

        def fake_create(model, messages, max_tokens, temperature=None, top_p=None, **kw):
            captured["temperature"] = temperature
            captured["top_p"] = top_p

        with patch.object(ocr_svc_custom, "_create_completion", side_effect=fake_create):
            ocr_svc_custom._call_ocr_api("gpt-4o", "system", "sys prompt", "user prompt", "data:image/png;base64,abc", 100)

        assert captured["temperature"] == pytest.approx(0.5)
        assert captured["top_p"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# PromptService — override vs. default
# ---------------------------------------------------------------------------

class TestPromptServiceSamplingOverride:

    def test_default_temperature_used_when_no_override(self, prompt_svc):
        captured = {}

        def fake_create(model, messages, max_tokens, temperature=None, top_p=None, **kw):
            captured["temperature"] = temperature
            captured["top_p"] = top_p

        with patch.object(prompt_svc, "_create_completion", side_effect=fake_create):
            prompt_svc._call_api("gpt-4o", "system", "sys prompt", "user prompt")

        assert captured["temperature"] == pytest.approx(PROMPT_TEMPERATURE)
        assert captured["top_p"] == pytest.approx(PROMPT_TOP_P)

    def test_custom_values_override_prompt_defaults(self, prompt_svc_custom):
        captured = {}

        def fake_create(model, messages, max_tokens, temperature=None, top_p=None, **kw):
            captured["temperature"] = temperature
            captured["top_p"] = top_p

        with patch.object(prompt_svc_custom, "_create_completion", side_effect=fake_create):
            prompt_svc_custom._call_api("gpt-4o", "system", "sys prompt", "user prompt")

        assert captured["temperature"] == pytest.approx(0.3)
        assert captured["top_p"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# ImageTranslationService — override vs. default
# ---------------------------------------------------------------------------

class TestImageTranslationServiceSamplingOverride:

    def test_default_temperature_used_when_no_override(self, img_trans_svc):
        captured = {}

        def fake_create(model, messages, max_tokens, temperature=None, **kw):
            captured["temperature"] = temperature

        with patch.object(img_trans_svc, "_create_completion", side_effect=fake_create):
            img_trans_svc._call_api(
                "gpt-4o", "system", "sys prompt", "user prompt",
                "data:image/png;base64,abc", 100,
            )

        assert captured["temperature"] == pytest.approx(IMAGE_TRANSLATION_TEMPERATURE)

    def test_custom_temperature_overrides_default(self, img_trans_svc_custom):
        captured = {}

        def fake_create(model, messages, max_tokens, temperature=None, **kw):
            captured["temperature"] = temperature

        with patch.object(img_trans_svc_custom, "_create_completion", side_effect=fake_create):
            img_trans_svc_custom._call_api(
                "gpt-4o", "system", "sys prompt", "user prompt",
                "data:image/png;base64,abc", 100,
            )

        assert captured["temperature"] == pytest.approx(0.1)
