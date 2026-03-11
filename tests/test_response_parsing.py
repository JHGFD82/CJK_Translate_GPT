"""
Tests for ImageTranslationService._parse_response.

_parse_response extracts [TRANSCRIPT] and [TRANSLATION] sections from the
model's raw text output.  It falls back to treating the full content as the
translation when neither header is present.
"""

import pytest


class TestParseResponse:
    """Unit tests for _parse_response — no API calls, pure regex logic."""

    def test_well_formed_response(self, image_translation_service):
        content = "[TRANSCRIPT]\nHello world\n\n[TRANSLATION]\nSekai"
        transcript, translation = image_translation_service._parse_response(content)
        assert transcript == "Hello world"
        assert translation == "Sekai"

    def test_multiline_sections(self, image_translation_service):
        content = (
            "[TRANSCRIPT]\n"
            "Line one\n"
            "Line two\n\n"
            "[TRANSLATION]\n"
            "Translated one\n"
            "Translated two"
        )
        transcript, translation = image_translation_service._parse_response(content)
        assert "Line one" in transcript
        assert "Line two" in transcript
        assert "Translated one" in translation
        assert "Translated two" in translation

    def test_whitespace_stripped_from_sections(self, image_translation_service):
        content = "[TRANSCRIPT]\n\n  Hello  \n\n[TRANSLATION]\n\n  World  \n\n"
        transcript, translation = image_translation_service._parse_response(content)
        assert transcript == "Hello"
        assert translation == "World"

    def test_only_translation_header_present(self, image_translation_service):
        # No [TRANSCRIPT] → transcript empty, translation populated
        content = "[TRANSLATION]\nOnly a translation here"
        transcript, translation = image_translation_service._parse_response(content)
        assert transcript == ""
        assert translation == "Only a translation here"

    def test_only_transcript_header_present(self, image_translation_service):
        # No [TRANSLATION] → translation empty, transcript populated; no fallback triggered
        content = "[TRANSCRIPT]\nOnly a transcript here"
        transcript, translation = image_translation_service._parse_response(content)
        assert transcript == "Only a transcript here"
        assert translation == ""

    def test_neither_header_falls_back_to_full_content(self, image_translation_service):
        # When both sections are empty the raw content becomes the translation
        content = "The model returned plain text with no section markers."
        transcript, translation = image_translation_service._parse_response(content)
        assert transcript == ""
        assert translation == content.strip()

    def test_empty_content_falls_back(self, image_translation_service):
        content = "   "
        transcript, translation = image_translation_service._parse_response(content)
        assert transcript == ""
        assert translation == ""

    def test_headers_case_sensitive(self, image_translation_service):
        # Headers are uppercase in spec; lowercase versions should NOT be recognised
        content = "[transcript]\nHello\n\n[translation]\nWorld"
        transcript, translation = image_translation_service._parse_response(content)
        # Falls back: neither section matched → full content becomes translation
        assert transcript == "" or "Hello" not in transcript

    def test_content_between_headers_not_leaked(self, image_translation_service):
        # Text before the first header should not appear in either section
        content = "Preamble text.\n\n[TRANSCRIPT]\nActual transcript\n\n[TRANSLATION]\nActual translation"
        transcript, translation = image_translation_service._parse_response(content)
        assert "Preamble" not in transcript
        assert "Preamble" not in translation

    def test_cjk_content_in_transcript(self, image_translation_service):
        content = "[TRANSCRIPT]\n日本語のテキスト\n\n[TRANSLATION]\nJapanese text"
        transcript, translation = image_translation_service._parse_response(content)
        assert "日本語のテキスト" in transcript
        assert "Japanese text" in translation
