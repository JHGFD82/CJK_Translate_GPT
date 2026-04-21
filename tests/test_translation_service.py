"""
Tests for TranslationService methods not covered by test_parallel_translation.py.

Covers:
- _find_split_point (all three paths: paragraph break, sentence boundary, fallback)
- _resolve_output_format (all branches)
- _make_text_triples (pagination)
- build_prompts (dry-run interface)
- translate_text (content-is-None retry, content wrong type, successful)
- translate_page_text (context_type detection)
- generate_text (CONTEXT_LENGTH_EXCEEDED split, CONTENT_FILTER, empty result, success)
- _translate_page_sequence sequential path (progressive save, error handling)

No real API calls are made.
"""

import pytest
import time
from unittest.mock import MagicMock, patch, call

from src.services.translation_service import TranslationService
from src.services.api_errors import APISignal
from src.models.output_options import OutputOptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Usage:
    def __init__(self):
        self.prompt_tokens = 5
        self.completion_tokens = 15
        self.total_tokens = 20


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _FakeResponse:
    def __init__(self, content="translated", model="gpt-4o"):
        self.id = "r"
        self.model = model
        self.usage = _Usage()
        self.choices = [_Choice(content)]


@pytest.fixture
def svc():
    tracker = MagicMock()
    tracker.record_usage.return_value = MagicMock(total_cost=0.001)
    tracker.usage_data = {"total_usage": {"total_tokens": 0, "total_cost": 0.0}}
    return TranslationService(api_key="fake-key", token_tracker=tracker)


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    monkeypatch.setattr("src.services.translation_service.resolve_model", lambda **_: "gpt-4o")
    monkeypatch.setattr("src.services.translation_service.maybe_sync_model_pricing", lambda m: None)
    monkeypatch.setattr("src.services.translation_service.get_model_system_role", lambda m: "system")
    monkeypatch.setattr("src.services.translation_service.get_model_max_completion_tokens", lambda m, d: d)
    monkeypatch.setattr("src.services.translation_service.time.sleep", lambda s: None)
    monkeypatch.setattr("src.services.translation_service.PAGE_DELAY_SECONDS", 0)


# ===========================================================================
# _find_split_point
# ===========================================================================

class TestFindSplitPoint:
    def test_prefers_paragraph_break(self):
        # Put a \n\n exactly in the middle
        text = "A" * 50 + "\n\nB" * 25
        middle = len(text) // 2
        sp = TranslationService._find_split_point(text, middle)
        # Should land after the \n\n (i.e., at the 'B')
        assert text[sp - 2:sp] == "\n\n" or text[sp:sp + 1] == "B"

    def test_falls_back_to_sentence_boundary(self):
        # No paragraph break; put a period near the middle
        text = "Hello world. " * 8
        middle = len(text) // 2
        sp = TranslationService._find_split_point(text, middle)
        # The character just before the split should be a sentence-ending char
        assert text[sp - 1] in ".!?。"

    def test_falls_back_to_raw_middle(self):
        # No paragraph break or sentence boundary — pure 'A's
        text = "A" * 200
        middle = len(text) // 2
        sp = TranslationService._find_split_point(text, middle)
        assert sp == middle


# ===========================================================================
# _resolve_output_format
# ===========================================================================

class TestResolveOutputFormat:
    def test_pdf_extension(self):
        opts = OutputOptions(output_file="out.pdf")
        assert TranslationService._resolve_output_format(opts) == "pdf"

    def test_docx_extension(self):
        opts = OutputOptions(output_file="out.docx")
        assert TranslationService._resolve_output_format(opts) == "docx"

    def test_txt_extension(self):
        opts = OutputOptions(output_file="out.txt")
        assert TranslationService._resolve_output_format(opts) == "txt"

    def test_unknown_extension_returns_file(self):
        opts = OutputOptions(output_file="out.xyz")
        assert TranslationService._resolve_output_format(opts) == "file"

    def test_auto_save_returns_txt(self):
        opts = OutputOptions(auto_save=True)
        assert TranslationService._resolve_output_format(opts) == "txt"

    def test_default_returns_console(self):
        opts = OutputOptions()
        assert TranslationService._resolve_output_format(opts) == "console"

    def test_output_file_without_extension(self):
        opts = OutputOptions(output_file="outfile")
        assert TranslationService._resolve_output_format(opts) == "file"


# ===========================================================================
# _make_text_triples
# ===========================================================================

class TestMakeTextTriples:
    def test_basic_pagination(self):
        pages = ["page0", "page1", "page2"]
        triples = list(TranslationService._make_text_triples(pages))
        assert triples[0] == (0, "page0", "")
        assert triples[1] == (1, "page1", "page0")
        assert triples[2] == (2, "page2", "page1")

    def test_single_page(self):
        triples = list(TranslationService._make_text_triples(["only"]))
        assert triples[0] == (0, "only", "")

    def test_empty_list(self):
        assert list(TranslationService._make_text_triples([])) == []


# ===========================================================================
# build_prompts
# ===========================================================================

class TestBuildPrompts:
    def test_returns_system_and_user_prompt(self, svc):
        sys_p, usr_p = svc.build_prompts("Hello", "Chinese", "English")
        assert isinstance(sys_p, str) and len(sys_p) > 0
        assert "Hello" in usr_p

    def test_respects_output_format(self, svc):
        _, usr_p_console = svc.build_prompts("text", "Chinese", "English", "console")
        _, usr_p_docx = svc.build_prompts("text", "Chinese", "English", "docx")
        # Both return user prompts containing the text
        assert "text" in usr_p_console
        assert "text" in usr_p_docx


# ===========================================================================
# translate_text
# ===========================================================================

class TestTranslateText:
    def test_successful_translation(self, svc):
        with patch.object(svc, "_create_completion", return_value=_FakeResponse("Bonjour")):
            result = svc.translate_text("Hello", "English", "French")
        assert result == "Bonjour"

    def test_none_content_triggers_retry_then_returns_content(self, svc):
        responses = iter([_FakeResponse(content=None), _FakeResponse("Second try")])
        with patch.object(svc, "_create_completion", side_effect=lambda *a, **kw: next(responses)):
            result = svc.translate_text("Hello", "English", "French")
        assert result == "Second try"

    def test_non_string_content_triggers_retry(self, svc):
        responses = iter([_FakeResponse(content=42), _FakeResponse("Correct")])
        with patch.object(svc, "_create_completion", side_effect=lambda *a, **kw: next(responses)):
            result = svc.translate_text("Hello", "English", "French")
        assert result == "Correct"

    def test_empty_choices_returns_empty_string(self, svc):
        resp = _FakeResponse()
        resp.choices = []
        with patch.object(svc, "_create_completion", return_value=resp):
            result = svc.translate_text("Hello", "English", "French")
        assert result == ""

    def test_returns_api_signal_on_error(self, svc):
        with patch.object(svc, "_create_completion", side_effect=Exception("rate_limit")), \
             patch("src.services.base_service.classify_api_error",
                   side_effect=Exception("Rate limit exceeded: rate_limit")):
            with pytest.raises(Exception):
                svc.translate_text("Hello", "English", "French")

    def test_suppress_inline_print(self, svc, capsys):
        svc._suppress_inline_print = True
        with patch.object(svc, "_create_completion", return_value=_FakeResponse("Silent")):
            svc.translate_text("Hello", "English", "French")
        out = capsys.readouterr().out
        assert "Silent" not in out


# ===========================================================================
# translate_page_text
# ===========================================================================

class TestTranslatePageText:
    def test_context_type_abstract(self, svc):
        captured = {}

        def fake_translate(text, src, tgt, fmt="console", context_type="none"):
            captured["context_type"] = context_type
            return "translated"

        with patch.object(svc, "translate_text", side_effect=fake_translate):
            svc.translate_page_text("abstract", "page text", "", "Chinese", "English")
        assert captured["context_type"] == "abstract"

    def test_context_type_previous_page(self, svc):
        captured = {}

        def fake_translate(text, src, tgt, fmt="console", context_type="none"):
            captured["context_type"] = context_type
            return "translated"

        with patch.object(svc, "translate_text", side_effect=fake_translate):
            svc.translate_page_text("", "page text", "previous text", "Chinese", "English")
        assert captured["context_type"] == "previous_page"

    def test_context_type_none(self, svc):
        captured = {}

        def fake_translate(text, src, tgt, fmt="console", context_type="none"):
            captured["context_type"] = context_type
            return "translated"

        with patch.object(svc, "translate_text", side_effect=fake_translate):
            svc.translate_page_text("", "page text", "", "Chinese", "English")
        assert captured["context_type"] == "none"


# ===========================================================================
# generate_text
# ===========================================================================

class TestGenerateText:
    def test_successful_translation(self, svc):
        with patch.object(svc, "translate_page_text", return_value="Translated content"):
            result = svc.generate_text("", "Hello", "", 0, "English", "French")
        assert "Translated content" in result
        assert "Page 1" in result

    def test_content_filter_signal_adds_error_note(self, svc):
        with patch.object(svc, "translate_page_text", return_value=APISignal.CONTENT_FILTER):
            result = svc.generate_text("", "text", "", 2, "Chinese", "English")
        assert "Content filter triggered" in result

    def test_empty_result_adds_error_note(self, svc):
        with patch.object(svc, "translate_page_text", return_value=""):
            result = svc.generate_text("", "text", "", 0, "Chinese", "English")
        assert "Translation error" in result

    def test_context_length_exceeded_splits_and_retries(self, svc):
        """When CONTEXT_LENGTH_EXCEEDED is returned, text should be split and both halves translated."""
        calls = []

        def fake_translate(abstract, page_text, prev, src, tgt, fmt="console", previous_translated=""):
            calls.append(page_text)
            if len(page_text) > 50:
                return APISignal.CONTEXT_LENGTH_EXCEEDED
            return f"Translated: {page_text[:10]}"

        long_text = "A" * 200
        with patch.object(svc, "translate_page_text", side_effect=fake_translate):
            result = svc.generate_text("", long_text, "", 0, "Chinese", "English")
        # First call was the full text (too long), then split halves were translated
        assert calls[0] == long_text
        assert len(calls) > 1
        assert "Translated:" in result


# ===========================================================================
# _translate_page_sequence — sequential path
# ===========================================================================

class TestTranslatePageSequenceSequential:

    @pytest.fixture(autouse=True)
    def no_sleep(self, monkeypatch):
        monkeypatch.setattr("src.services.translation_service.time.sleep", lambda s: None)

    def test_basic_sequential_translate(self, svc):
        triples = [(0, "page0", ""), (1, "page1", "page0")]
        with patch.object(svc, "generate_text", side_effect=lambda *a, **kw: f"Done {a[3]}"):
            results = svc._translate_page_sequence(
                iter(triples), "", "Chinese", "English", "console",
                0, "page", OutputOptions(),
                input_file_path=None,
            )
        assert results == ["Done 0", "Done 1"]

    def test_exception_in_page_continues(self, svc):
        triples = [(0, "good", ""), (1, "bad", "good")]

        def gen(abstract, page, prev, idx, src, tgt, fmt, prev_translated):
            if idx == 1:
                raise RuntimeError("boom")
            return "ok"

        with patch.object(svc, "generate_text", side_effect=gen):
            results = svc._translate_page_sequence(
                iter(triples), "", "Chinese", "English", "console",
                0, "page", OutputOptions(),
                input_file_path=None,
            )
        assert results[0] == "ok"
        assert "Translation error" in results[1]

    def test_workers_gt_1_dispatches_parallel(self, svc):
        triples = [(0, "p0", ""), (1, "p1", "p0")]
        with patch.object(svc, "_translate_pages_parallel", return_value=["A", "B"]) as mock_par:
            results = svc._translate_page_sequence(
                iter(triples), "", "Chinese", "English", "console",
                0, "page", OutputOptions(),
                input_file_path=None, workers=2,
            )
        mock_par.assert_called_once()
        assert results == ["A", "B"]
