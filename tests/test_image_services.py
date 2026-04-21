"""
Tests for ImageTranslationService and ImageProcessorService.

Covers:
- ImageTranslationService._get_model, _get_max_tokens, build_prompts,
  _parse_response, process_image_translation
- ImageProcessorService._get_model, build_prompts, _build_refinement_prompt,
  process_image_ocr (basic, multi-pass, vision-check failure, image-load failure)
  _run_single_refinement_pass

No real API calls or file I/O are made.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.services.image_translation_service import ImageTranslationService
from src.services.image_processor_service import ImageProcessorService


# ---------------------------------------------------------------------------
# Shared minimal response stand-in (non-iterable)
# ---------------------------------------------------------------------------

class _Usage:
    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 20
        self.total_tokens = 30


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _FakeResponse:
    def __init__(self, content="result", model="gpt-4o"):
        self.id = "r1"
        self.model = model
        self.usage = _Usage()
        self.choices = [_Choice(content)]


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_catalog_funcs(monkeypatch):
    """Prevent all catalog/disk access in service helpers."""
    monkeypatch.setattr("src.services.image_translation_service.resolve_model", lambda **_: "gpt-4o")
    monkeypatch.setattr("src.services.image_translation_service.maybe_sync_model_pricing", lambda m: None)
    monkeypatch.setattr("src.services.image_translation_service.get_model_system_role", lambda m: "system")
    monkeypatch.setattr("src.services.image_translation_service.get_model_max_completion_tokens", lambda m, d: d)
    monkeypatch.setattr("src.services.image_translation_service.model_supports_vision", lambda m: True)
    monkeypatch.setattr("src.services.image_translation_service.get_default_model", lambda r: "gpt-4o")

    monkeypatch.setattr("src.services.image_processor_service.resolve_model", lambda **_: "gpt-4o")
    monkeypatch.setattr("src.services.image_processor_service.maybe_sync_model_pricing", lambda m: None)
    monkeypatch.setattr("src.services.image_processor_service.get_model_system_role", lambda m: "system")
    monkeypatch.setattr("src.services.image_processor_service.get_model_max_completion_tokens", lambda m, d: d)
    monkeypatch.setattr("src.services.image_processor_service.model_supports_vision", lambda m: True)
    monkeypatch.setattr("src.services.image_processor_service.get_default_model", lambda r: "gpt-4o-mini")
    monkeypatch.setattr("src.services.image_processor_service.get_vision_capable_models", lambda: ["gpt-4o"])


@pytest.fixture
def img_trans_svc():
    tracker = MagicMock()
    tracker.record_usage.return_value = MagicMock(total_cost=0.0)
    return ImageTranslationService(api_key="fake-key", token_tracker=tracker)


@pytest.fixture
def ocr_svc():
    tracker = MagicMock()
    tracker.record_usage.return_value = MagicMock(total_cost=0.0)
    return ImageProcessorService(api_key="fake-key", token_tracker=tracker)


# ===========================================================================
# ImageTranslationService
# ===========================================================================

class TestImageTranslationServiceGetModel:
    def test_returns_resolved_model(self, img_trans_svc):
        assert img_trans_svc._get_model() == "gpt-4o"

    def test_warns_when_preferred_model_differs(self, img_trans_svc, monkeypatch, caplog):
        import logging
        monkeypatch.setattr(
            "src.services.image_translation_service.resolve_model", lambda **_: "gpt-4o-mini"
        )
        with caplog.at_level(logging.WARNING):
            model = img_trans_svc._get_model()
        assert model == "gpt-4o-mini"
        assert any("not available" in r.message.lower() for r in caplog.records)


class TestImageTranslationServiceGetMaxTokens:
    def test_custom_max_tokens_used(self):
        svc = ImageTranslationService(api_key="fake", token_tracker=MagicMock(), max_tokens=1234)
        assert svc._get_max_tokens("gpt-4o") == 1234

    def test_default_max_tokens_from_catalog(self, img_trans_svc, monkeypatch):
        monkeypatch.setattr(
            "src.services.image_translation_service.get_model_max_completion_tokens",
            lambda m, d: 9999,
        )
        assert img_trans_svc._get_max_tokens("gpt-4o") == 9999


class TestImageTranslationServiceBuildPrompts:
    def test_returns_system_and_user_prompts(self, img_trans_svc):
        sys_p, usr_p = img_trans_svc.build_prompts("Chinese", "English")
        assert isinstance(sys_p, str) and len(sys_p) > 0
        assert isinstance(usr_p, str) and len(usr_p) > 0

    def test_system_note_injected(self, img_trans_svc):
        img_trans_svc.system_note = "Extra instruction"
        sys_p, _ = img_trans_svc.build_prompts("Chinese", "English")
        assert "Extra instruction" in sys_p

    def test_user_note_injected(self, img_trans_svc):
        img_trans_svc.user_note = "User guidance"
        _, usr_p = img_trans_svc.build_prompts("Chinese", "English")
        assert "User guidance" in usr_p

    def test_vertical_flag_forwarded(self, img_trans_svc):
        _, usr_p_h = img_trans_svc.build_prompts("Japanese", "English", vertical=False)
        _, usr_p_v = img_trans_svc.build_prompts("Japanese", "English", vertical=True)
        # Vertical prompt should differ from horizontal
        assert usr_p_h != usr_p_v


class TestParseResponse:
    def test_well_formed_sections(self, img_trans_svc):
        content = "[TRANSCRIPT]\nOriginal text\n[TRANSLATION]\nTranslated text"
        transcript, translation = img_trans_svc._parse_response(content)
        assert transcript == "Original text"
        assert translation == "Translated text"

    def test_missing_both_sections_uses_full_response(self, img_trans_svc):
        content = "Just a plain response"
        transcript, translation = img_trans_svc._parse_response(content)
        assert transcript == ""
        assert translation == "Just a plain response"

    def test_transcript_only_gives_empty_translation(self, img_trans_svc):
        content = "[TRANSCRIPT]\nOnly transcript"
        transcript, translation = img_trans_svc._parse_response(content)
        assert transcript == "Only transcript"
        assert translation == ""

    def test_translation_only_gives_empty_transcript(self, img_trans_svc):
        content = "[TRANSLATION]\nOnly translation"
        transcript, translation = img_trans_svc._parse_response(content)
        assert transcript == ""
        assert translation == "Only translation"


class TestProcessImageTranslation:
    def test_successful_call(self, img_trans_svc):
        content = "[TRANSCRIPT]\nOriginal\n[TRANSLATION]\nTranslated"
        with patch.object(img_trans_svc.image_processor, "local_image_to_data_url", return_value="data:image/png;base64,abc"), \
             patch.object(img_trans_svc, "_create_completion", return_value=_FakeResponse(content)):
            transcript, translation = img_trans_svc.process_image_translation(
                "fake.jpg", "Chinese", "English"
            )
        assert transcript == "Original"
        assert translation == "Translated"

    def test_vision_check_failure_raises(self, img_trans_svc, monkeypatch):
        monkeypatch.setattr(
            "src.services.image_translation_service.model_supports_vision", lambda m: False
        )
        with pytest.raises(ValueError, match="does not support image"):
            img_trans_svc.process_image_translation("fake.jpg", "Chinese", "English")

    def test_image_load_failure_raises(self, img_trans_svc):
        with patch.object(img_trans_svc.image_processor, "local_image_to_data_url",
                          side_effect=FileNotFoundError("no such file")):
            with pytest.raises(FileNotFoundError):
                img_trans_svc.process_image_translation("missing.jpg", "Chinese", "English")

    def test_none_content_triggers_retry(self, img_trans_svc):
        responses = iter([_FakeResponse(content=None), _FakeResponse("[TRANSCRIPT]\nR\n[TRANSLATION]\nT")])
        with patch.object(img_trans_svc.image_processor, "local_image_to_data_url", return_value="data:url"), \
             patch.object(img_trans_svc, "_create_completion",
                          side_effect=lambda *a, **kw: next(responses)):
            transcript, translation = img_trans_svc.process_image_translation(
                "img.jpg", "Chinese", "English"
            )
        assert translation == "T"


# ===========================================================================
# ImageProcessorService
# ===========================================================================

class TestImageProcessorServiceGetModel:
    def test_returns_resolved_model(self, ocr_svc):
        assert ocr_svc._get_model() == "gpt-4o"

    def test_warns_when_fallback_used(self, ocr_svc, monkeypatch, caplog):
        import logging
        monkeypatch.setattr(
            "src.services.image_processor_service.resolve_model", lambda **_: "gpt-4o"
        )
        # default model is "gpt-4o-mini" but resolved is "gpt-4o" → warning
        with caplog.at_level(logging.WARNING):
            model = ocr_svc._get_model()
        assert model == "gpt-4o"
        assert any("not available" in r.message for r in caplog.records)


class TestImageProcessorServiceBuildPrompts:
    def test_build_prompts_returns_strings(self, ocr_svc):
        sys_p, usr_p = ocr_svc.build_prompts("English")
        assert isinstance(sys_p, str) and isinstance(usr_p, str)

    def test_refinement_prompt_returns_string(self, ocr_svc):
        rp = ocr_svc._build_refinement_prompt("Japanese")
        assert isinstance(rp, str) and len(rp) > 0

    def test_vertical_flag_changes_prompt(self, ocr_svc):
        _, usr_h = ocr_svc.build_prompts("Japanese", vertical=False)
        _, usr_v = ocr_svc.build_prompts("Japanese", vertical=True)
        assert usr_h != usr_v


class TestProcessImageOcr:
    def test_basic_ocr(self, ocr_svc):
        with patch.object(ocr_svc.image_processor, "local_image_to_data_url", return_value="data:url"), \
             patch.object(ocr_svc, "_create_completion", return_value=_FakeResponse("Extracted text")):
            result = ocr_svc.process_image_ocr("img.jpg", "English")
        assert result == "Extracted text"

    def test_vision_check_failure_raises(self, ocr_svc, monkeypatch):
        monkeypatch.setattr(
            "src.services.image_processor_service.model_supports_vision", lambda m: False
        )
        with pytest.raises(ValueError, match="does not support image"):
            ocr_svc.process_image_ocr("img.jpg", "English")

    def test_image_load_failure_raises(self, ocr_svc):
        with patch.object(ocr_svc.image_processor, "local_image_to_data_url",
                          side_effect=OSError("permission denied")):
            with pytest.raises(OSError):
                ocr_svc.process_image_ocr("img.jpg", "English")

    def test_none_content_triggers_retry(self, ocr_svc):
        responses = iter([_FakeResponse(content=None), _FakeResponse("Second pass")])
        with patch.object(ocr_svc.image_processor, "local_image_to_data_url", return_value="data:url"), \
             patch.object(ocr_svc, "_create_completion",
                          side_effect=lambda *a, **kw: next(responses)):
            result = ocr_svc.process_image_ocr("img.jpg", "English")
        assert result == "Second pass"

    def test_empty_content_triggers_retry(self, ocr_svc):
        responses = iter([_FakeResponse(content="   "), _FakeResponse("Good result")])
        with patch.object(ocr_svc.image_processor, "local_image_to_data_url", return_value="data:url"), \
             patch.object(ocr_svc, "_create_completion",
                          side_effect=lambda *a, **kw: next(responses)):
            result = ocr_svc.process_image_ocr("img.jpg", "English")
        assert result == "Good result"

    def test_multi_pass_calls_refinement(self, ocr_svc):
        with patch.object(ocr_svc.image_processor, "local_image_to_data_url", return_value="data:url"), \
             patch.object(ocr_svc, "_create_completion", return_value=_FakeResponse("Pass 1")), \
             patch.object(ocr_svc, "_run_single_refinement_pass",
                          return_value="Refined") as mock_refine:
            result = ocr_svc.process_image_ocr("img.jpg", "English", passes=2)
        mock_refine.assert_called_once()
        assert result == "Refined"


class TestRunSingleRefinementPass:
    def test_successful_refinement(self, ocr_svc):
        with patch.object(ocr_svc, "_create_completion", return_value=_FakeResponse("Refined text")):
            result = ocr_svc._run_single_refinement_pass(
                "gpt-4o", "system", "sys prompt",
                "user prompt", "data:url", "prior transcription",
                "refinement prompt", 4000, 2,
            )
        assert result == "Refined text"

    def test_none_content_triggers_retry(self, ocr_svc):
        responses = iter([_FakeResponse(content=None), _FakeResponse("Retry ok")])
        with patch.object(ocr_svc, "_create_completion",
                          side_effect=lambda *a, **kw: next(responses)):
            result = ocr_svc._run_single_refinement_pass(
                "gpt-4o", "system", "sys prompt",
                "user prompt", "data:url", "prior",
                "refine prompt", 4000, 2,
            )
        assert result == "Retry ok"

    def test_empty_content_triggers_retry(self, ocr_svc):
        responses = iter([_FakeResponse(content="  "), _FakeResponse("Non-empty")])
        with patch.object(ocr_svc, "_create_completion",
                          side_effect=lambda *a, **kw: next(responses)):
            result = ocr_svc._run_single_refinement_pass(
                "gpt-4o", "system", "sys", "user", "data:url", "prior", "refine", 4000, 2,
            )
        assert result == "Non-empty"
