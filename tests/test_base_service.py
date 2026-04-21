"""
Tests for BaseService — init, _create_completion, _record_response_usage,
_run_with_retry, and sampling-parameter override logic.

No API calls are made; the Portkey client and external functions are patched.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from src.services.base_service import BaseService
from src.services.api_errors import APISignal
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


# ===========================================================================
# BaseService.__init__
# ===========================================================================

def _make_svc(**kwargs) -> BaseService:
    """Instantiate TranslationService (a concrete BaseService subclass) with fake keys."""
    return TranslationService(api_key="fake-key", **kwargs)


class TestBaseServiceInitExtended:

    def test_api_key_stored(self):
        svc = _make_svc()
        assert svc.api_key == "fake-key"

    def test_professor_stored(self):
        svc = _make_svc(professor="heller")
        assert svc.professor == "heller"

    def test_professor_defaults_to_none(self):
        svc = _make_svc()
        assert svc.professor is None

    def test_custom_model_stored(self):
        svc = _make_svc(model="gpt-4o-mini")
        assert svc.custom_model == "gpt-4o-mini"

    def test_custom_model_defaults_to_none(self):
        svc = _make_svc()
        assert svc.custom_model is None

    def test_custom_max_tokens_stored(self):
        svc = _make_svc(max_tokens=999)
        assert svc.custom_max_tokens == 999

    def test_system_note_initialised_to_none(self):
        svc = _make_svc()
        assert svc.system_note is None

    def test_user_note_initialised_to_none(self):
        svc = _make_svc()
        assert svc.user_note is None

    def test_provided_token_tracker_reused(self):
        tracker = MagicMock()
        svc = _make_svc(token_tracker=tracker)
        assert svc.token_tracker is tracker

    def test_token_tracker_created_when_not_provided(self):
        """A fresh TokenTracker should be created when none is supplied."""
        with patch("src.services.base_service.TokenTracker") as MockTracker:
            MockTracker.return_value = MagicMock()
            svc = _make_svc(professor="heller")
            MockTracker.assert_called_once()
            assert svc.token_tracker is MockTracker.return_value


# ===========================================================================
# BaseService._create_completion — three API branches
# ===========================================================================

class TestCreateCompletion:
    """Verify the correct Portkey kwargs for each model type."""

    MESSAGES = [{"role": "user", "content": "hi"}]

    def _svc(self) -> BaseService:
        return _make_svc(token_tracker=MagicMock())

    def test_regular_model_uses_max_tokens(self):
        svc = self._svc()
        with patch("src.services.base_service.model_uses_max_completion_tokens", return_value=False), \
             patch("src.services.base_service.model_has_fixed_parameters", return_value=False), \
             patch.object(svc.client.chat.completions, "create", return_value=MagicMock()) as mock_create:
            svc._create_completion("gpt-4o", self.MESSAGES, 100, temperature=0.5, top_p=0.9)
            kwargs = mock_create.call_args.kwargs
            assert "max_tokens" in kwargs
            assert "max_completion_tokens" not in kwargs
            assert kwargs["temperature"] == 0.5
            assert kwargs["top_p"] == 0.9

    def test_reasoning_model_uses_max_completion_tokens(self):
        svc = self._svc()
        with patch("src.services.base_service.model_uses_max_completion_tokens", return_value=True), \
             patch("src.services.base_service.model_has_fixed_parameters", return_value=False), \
             patch.object(svc.client.chat.completions, "create", return_value=MagicMock()) as mock_create:
            svc._create_completion("o1", self.MESSAGES, 200, temperature=0.5, top_p=0.9)
            kwargs = mock_create.call_args.kwargs
            assert "max_completion_tokens" in kwargs
            assert "max_tokens" not in kwargs
            assert kwargs["temperature"] == 0.5

    def test_fixed_params_model_strips_temperature_and_top_p(self):
        svc = self._svc()
        with patch("src.services.base_service.model_uses_max_completion_tokens", return_value=True), \
             patch("src.services.base_service.model_has_fixed_parameters", return_value=True), \
             patch.object(svc.client.chat.completions, "create", return_value=MagicMock()) as mock_create:
            svc._create_completion("o1-mini", self.MESSAGES, 50, temperature=0.5, top_p=0.9)
            kwargs = mock_create.call_args.kwargs
            assert "temperature" not in kwargs
            assert "top_p" not in kwargs
            assert "max_completion_tokens" in kwargs

    def test_extra_kwargs_forwarded(self):
        svc = self._svc()
        with patch("src.services.base_service.model_uses_max_completion_tokens", return_value=False), \
             patch("src.services.base_service.model_has_fixed_parameters", return_value=False), \
             patch.object(svc.client.chat.completions, "create", return_value=MagicMock()) as mock_create:
            svc._create_completion("gpt-4o", self.MESSAGES, 100, frequency_penalty=0.5)
            kwargs = mock_create.call_args.kwargs
            assert kwargs.get("frequency_penalty") == 0.5

    def test_none_temperature_not_forwarded(self):
        svc = self._svc()
        with patch("src.services.base_service.model_uses_max_completion_tokens", return_value=False), \
             patch("src.services.base_service.model_has_fixed_parameters", return_value=False), \
             patch.object(svc.client.chat.completions, "create", return_value=MagicMock()) as mock_create:
            svc._create_completion("gpt-4o", self.MESSAGES, 100, temperature=None, top_p=None)
            kwargs = mock_create.call_args.kwargs
            assert "temperature" not in kwargs
            assert "top_p" not in kwargs


# ===========================================================================
# BaseService._record_response_usage
# ===========================================================================

class _FakeUsage:
    def __init__(self, prompt, completion, total):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class _FakeResponse:
    """Minimal non-iterable API response stand-in."""
    def __init__(self, *, prompt=10, completion=20, total=30, model="gpt-4o", rid="resp-1"):
        self.id = rid
        self.model = model
        self.usage = _FakeUsage(prompt, completion, total) if prompt is not None else None


def _make_response(**kwargs):
    return _FakeResponse(**kwargs)


class TestRecordResponseUsage:

    def test_calls_token_tracker_with_correct_args(self):
        tracker = MagicMock()
        tracker.record_usage.return_value = MagicMock(total_cost=0.001)
        svc = _make_svc(token_tracker=tracker)
        resp = _make_response(prompt=5, completion=15, total=20, model="gpt-4o")
        svc._record_response_usage(resp, "gpt-4o")
        tracker.record_usage.assert_called_once_with(
            model="gpt-4o",
            prompt_tokens=5,
            completion_tokens=15,
            total_tokens=20,
            requested_model="gpt-4o",
        )

    def test_uses_response_model_over_fallback(self):
        tracker = MagicMock()
        tracker.record_usage.return_value = MagicMock(total_cost=0.0)
        svc = _make_svc(token_tracker=tracker)
        resp = _make_response(model="gpt-4o-mini")
        svc._record_response_usage(resp, "requested-model")
        assert tracker.record_usage.call_args.kwargs["model"] == "gpt-4o-mini"
        assert tracker.record_usage.call_args.kwargs["requested_model"] == "requested-model"

    def test_missing_usage_logs_warning(self, caplog):
        import logging
        svc = _make_svc(token_tracker=MagicMock())
        resp = _make_response(prompt=None)
        with caplog.at_level(logging.WARNING):
            svc._record_response_usage(resp, "gpt-4o")
        assert any("No token usage" in r.message for r in caplog.records)

    def test_missing_usage_critical_logs_error(self, caplog):
        import logging
        svc = _make_svc(token_tracker=MagicMock())
        resp = _make_response(prompt=None)
        with caplog.at_level(logging.ERROR):
            svc._record_response_usage(resp, "gpt-4o", critical=True)
        assert any("CRITICAL" in r.message for r in caplog.records)

    def test_missing_usage_no_token_tracker_call(self):
        tracker = MagicMock()
        svc = _make_svc(token_tracker=tracker)
        resp = _make_response(prompt=None)
        svc._record_response_usage(resp, "gpt-4o")
        tracker.record_usage.assert_not_called()


# ===========================================================================
# BaseService._run_with_retry
# ===========================================================================

class TestRunWithRetry:
    """Patch time.sleep to avoid real delays."""

    def _svc(self) -> BaseService:
        return _make_svc(token_tracker=MagicMock())

    @pytest.fixture(autouse=True)
    def no_sleep(self):
        with patch("src.services.base_service.time.sleep"):
            yield

    def test_immediate_success_returns_result(self):
        svc = self._svc()
        result = svc._run_with_retry(lambda attempt: "ok", model="gpt-4o")
        assert result == "ok"

    def test_none_then_success_retries(self):
        svc = self._svc()
        calls = []
        def body(attempt):
            calls.append(attempt)
            return None if attempt == 0 else "done"
        result = svc._run_with_retry(body, model="gpt-4o")
        assert result == "done"
        assert len(calls) == 2

    def test_all_none_raises_runtime_error(self):
        svc = self._svc()
        with pytest.raises(RuntimeError, match="returned no content"):
            svc._run_with_retry(lambda attempt: None, model="gpt-4o")

    def test_custom_timeout_msg(self):
        svc = self._svc()
        with pytest.raises(RuntimeError, match="my custom message"):
            svc._run_with_retry(lambda attempt: None, model="gpt-4o", timeout_msg="my custom message")

    def test_all_none_return_signal_returns_content_filter(self):
        svc = self._svc()
        result = svc._run_with_retry(lambda attempt: None, model="gpt-4o", return_signal_on_error=True)
        assert result == APISignal.CONTENT_FILTER

    def test_transient_error_retried(self):
        svc = self._svc()
        calls = []
        def body(attempt):
            calls.append(attempt)
            if attempt == 0:
                raise Exception("503 unavailable")
            return "recovered"
        result = svc._run_with_retry(body, model="gpt-4o")
        assert result == "recovered"
        assert len(calls) == 2

    def test_non_retryable_error_propagates(self):
        svc = self._svc()
        with patch("src.services.base_service.classify_api_error", side_effect=ValueError("bad model")):
            with pytest.raises(ValueError, match="bad model"):
                svc._run_with_retry(lambda attempt: (_ for _ in ()).throw(Exception("auth failed")), model="m")

    def test_non_retryable_error_returns_signal_when_flag_set(self):
        svc = self._svc()
        signal = APISignal.CONTEXT_LENGTH_EXCEEDED

        def body(attempt):
            raise Exception("context_length_exceeded")

        with patch("src.services.base_service.classify_api_error", return_value=signal):
            result = svc._run_with_retry(body, model="gpt-4o", return_signal_on_error=True)
        assert result is signal

    def test_content_filter_retried_then_returns_signal(self):
        """Content-filter on every attempt should exhaust retries → CONTENT_FILTER signal."""
        svc = self._svc()

        def body(attempt):
            raise Exception("content_filter blocked")

        with patch("src.services.base_service.classify_api_error", return_value=APISignal.CONTENT_FILTER):
            result = svc._run_with_retry(body, model="gpt-4o", return_signal_on_error=True)
        assert result == APISignal.CONTENT_FILTER
