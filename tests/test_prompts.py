"""
Tests for prompt construction across all three services.

Methods now delegate to the spec classes in src/services/prompts/.
Tests that previously called private service helpers (_build_system_prompt, etc.)
now use the corresponding spec classes directly so the behavioural assertions
are preserved without depending on the service's internal structure.

No API calls are made; all code under test is pure string building.
"""

import pytest

from src.services.prompts import OcrPromptSpec, ImageTranslationPromptSpec, TranslationPromptSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains(text: str, *fragments: str) -> bool:
    """Return True only if every fragment appears in text."""
    return all(f in text for f in fragments)


# ===========================================================================
# OcrPromptSpec system-prompt tests
# ===========================================================================

class TestOCRSystemPrompt:
    """OcrPromptSpec.system_prompt()"""

    def test_always_contains_rules_section(self):
        result = OcrPromptSpec("Japanese").system_prompt()
        assert "RULES:" in result

    def test_always_contains_do_not_translate(self):
        result = OcrPromptSpec("Japanese").system_prompt()
        assert "Do NOT translate" in result or "do NOT translate" in result.lower()

    def test_no_vertical_section_by_default(self):
        result = OcrPromptSpec("Japanese").system_prompt()
        assert "TEXT ORIENTATION" not in result
        assert "top-to-bottom" not in result
        assert "right-to-left" not in result

    def test_vertical_section_present_when_flag_set(self):
        result = OcrPromptSpec("Japanese", vertical=True).system_prompt()
        assert "TEXT ORIENTATION" in result
        assert "top-to-bottom" in result
        assert "right-to-left" in result

    def test_vertical_describes_column_order(self):
        result = OcrPromptSpec("Japanese", vertical=True).system_prompt()
        assert "rightmost column" in result

    def test_japanese_script_guidance_injected(self):
        result = OcrPromptSpec("Japanese").system_prompt()
        assert "SCRIPT NOTES:" in result
        assert "Japanese script" in result
        assert "hiragana" in result

    def test_english_script_guidance_injected(self):
        result = OcrPromptSpec("English").system_prompt()
        assert "SCRIPT NOTES:" in result
        assert "Latin alphabet" in result

    def test_korean_script_guidance_injected(self):
        result = OcrPromptSpec("Korean").system_prompt()
        assert "SCRIPT NOTES:" in result
        assert "hangul" in result

    def test_simplified_chinese_script_guidance(self):
        result = OcrPromptSpec("Simplified Chinese").system_prompt()
        assert "Simplified Chinese" in result
        assert "简体字" in result

    def test_traditional_chinese_script_guidance(self):
        result = OcrPromptSpec("Traditional Chinese").system_prompt()
        assert "Traditional Chinese" in result
        assert "繁體字" in result

    def test_unknown_language_has_no_script_notes(self):
        result = OcrPromptSpec("Spanish").system_prompt()
        assert "SCRIPT NOTES:" not in result

    def test_vertical_and_script_notes_both_present(self):
        result = OcrPromptSpec("Japanese", vertical=True).system_prompt()
        assert "SCRIPT NOTES:" in result
        assert "TEXT ORIENTATION" in result


class TestOCRUserPrompt:
    """OcrPromptSpec.user_prompt()"""

    def test_contains_target_language(self):
        result = OcrPromptSpec("Japanese").user_prompt()
        assert "Japanese" in result

    def test_contains_do_not_translate(self):
        result = OcrPromptSpec("Japanese").user_prompt()
        assert "Do NOT translate" in result

    def test_no_vertical_note_by_default(self):
        result = OcrPromptSpec("Japanese").user_prompt()
        assert "vertical" not in result

    def test_vertical_note_added_when_flag_set(self):
        result = OcrPromptSpec("Japanese", vertical=True).user_prompt()
        assert "vertical" in result
        assert "top-to-bottom" in result
        assert "right-to-left" in result

    def test_different_language_reflected(self):
        result = OcrPromptSpec("Korean").user_prompt()
        assert "Korean" in result
        assert "Japanese" not in result


class TestCreateOCRPrompt:
    """_create_ocr_prompt delegates to OcrPromptSpec; verify tuple structure."""

    def test_returns_two_strings(self, ocr_service):
        system, user = ocr_service._create_ocr_prompt("Japanese")
        assert isinstance(system, str) and isinstance(user, str)

    def test_vertical_propagates_to_both(self, ocr_service):
        system, user = ocr_service._create_ocr_prompt("Japanese", vertical=True)
        assert "TEXT ORIENTATION" in system
        assert "vertical" in user

    def test_non_vertical_has_no_orientation_in_either(self, ocr_service):
        system, user = ocr_service._create_ocr_prompt("Japanese", vertical=False)
        assert "TEXT ORIENTATION" not in system
        assert "vertical" not in user


# ===========================================================================
# ImageTranslationPromptSpec & service wrappers
# ===========================================================================

class TestImageTranslationSystemPrompt:
    """ImageTranslationPromptSpec.system_prompt() (via service wrapper)."""

    def test_contains_source_language(self, image_translation_service):
        result = image_translation_service._build_system_prompt("Japanese", "English")
        assert "Japanese" in result

    def test_contains_target_language(self, image_translation_service):
        result = image_translation_service._build_system_prompt("Japanese", "English")
        assert "English" in result

    def test_always_contains_transcript_and_translation_headers(self, image_translation_service):
        result = image_translation_service._build_system_prompt("Japanese", "English")
        assert "[TRANSCRIPT]" in result
        assert "[TRANSLATION]" in result

    def test_no_vertical_section_by_default(self, image_translation_service):
        result = image_translation_service._build_system_prompt("Japanese", "English")
        assert "TEXT ORIENTATION" not in result

    def test_vertical_section_present_when_flag_set(self, image_translation_service):
        result = image_translation_service._build_system_prompt("Japanese", "English", vertical=True)
        assert "TEXT ORIENTATION" in result
        assert "top-to-bottom" in result
        assert "right-to-left" in result

    def test_japanese_script_guidance_injected(self, image_translation_service):
        result = image_translation_service._build_system_prompt("Japanese", "English")
        assert "SCRIPT NOTES:" in result
        assert "Japanese script" in result

    def test_unknown_source_language_no_script_notes(self, image_translation_service):
        result = image_translation_service._build_system_prompt("Spanish", "English")
        assert "SCRIPT NOTES:" not in result

    def test_vertical_and_script_notes_coexist(self, image_translation_service):
        result = image_translation_service._build_system_prompt("Japanese", "English", vertical=True)
        assert "SCRIPT NOTES:" in result
        assert "TEXT ORIENTATION" in result


class TestImageTranslationUserPrompt:
    """_build_user_prompt for the combined OCR + translation service."""

    def test_contains_source_language(self, image_translation_service):
        result = image_translation_service._build_user_prompt("Japanese", "English")
        assert "Japanese" in result

    def test_contains_target_language(self, image_translation_service):
        result = image_translation_service._build_user_prompt("Japanese", "English")
        assert "English" in result

    def test_no_vertical_note_by_default(self, image_translation_service):
        result = image_translation_service._build_user_prompt("Japanese", "English")
        assert "vertical" not in result

    def test_vertical_note_added_when_flag_set(self, image_translation_service):
        result = image_translation_service._build_user_prompt("Japanese", "English", vertical=True)
        assert "vertical" in result
        assert "top-to-bottom" in result
        assert "right-to-left" in result

    def test_mentions_translation(self, image_translation_service):
        result = image_translation_service._build_user_prompt("Japanese", "English")
        assert "translate" in result


# ===========================================================================
# TranslationPromptSpec tests
# ===========================================================================

class TestFormattingInstruction:
    """TranslationPromptSpec.system_prompt() varies by output_format."""

    @pytest.mark.parametrize("fmt", ["pdf", "PDF", "txt", "TXT", "docx", "DOCX", "file", "FILE"])
    def test_file_formats_embed_file_instruction(self, fmt):
        result = TranslationPromptSpec("Japanese", "English", output_format=fmt).system_prompt()
        assert "file output" in result
        assert "actual line breaks" in result

    @pytest.mark.parametrize("fmt", ["console", "CONSOLE", "", "unknown", "screen"])
    def test_non_file_formats_embed_console_instruction(self, fmt):
        result = TranslationPromptSpec("Japanese", "English", output_format=fmt).system_prompt()
        assert "console" in result.lower()


class TestTranslationSystemPrompt:
    """TranslationPromptSpec.system_prompt()"""

    def test_contains_source_language(self):
        result = TranslationPromptSpec("Japanese", "English").system_prompt()
        assert "Japanese" in result

    def test_contains_target_language(self):
        result = TranslationPromptSpec("Japanese", "English").system_prompt()
        assert "English" in result

    def test_numbered_block_embedded_when_has_numbered(self):
        result = TranslationPromptSpec("Japanese", "English", has_numbered=True).system_prompt()
        assert "numbered lists" in result or "IMPORTANT" in result

    def test_numbered_block_absent_when_not_has_numbered(self):
        result = TranslationPromptSpec("Japanese", "English", has_numbered=False).system_prompt()
        assert "numbered lists" not in result

    def test_mentions_translation_role(self):
        result = TranslationPromptSpec("Japanese", "English").system_prompt()
        assert "professional translator" in result


class TestTranslationUserPromptTemplate:
    """TranslationPromptSpec.user_prompt()"""

    def test_contains_source_language(self):
        result = TranslationPromptSpec("Japanese", "English").user_prompt()
        assert "Japanese" in result

    def test_contains_target_language(self):
        result = TranslationPromptSpec("Japanese", "English").user_prompt()
        assert "English" in result

    def test_references_current_page_marker(self):
        result = TranslationPromptSpec("Japanese", "English").user_prompt()
        assert "--Current Page:" in result

    def test_mentions_numbering_preservation(self):
        result = TranslationPromptSpec("Japanese", "English", has_numbered=True).user_prompt()
        assert "numbering" in result.lower() or "CRITICAL" in result

    def test_no_numbering_instructions_when_not_detected(self):
        result = TranslationPromptSpec("Japanese", "English", has_numbered=False).user_prompt()
        assert "CRITICAL" not in result
        assert "NUMBERING CONTINUATION" not in result


class TestCreateTranslationPrompt:
    """_create_translation_prompt delegates to TranslationPromptSpec; verify tuple structure."""

    def test_returns_two_strings(self, translation_service):
        system, user = translation_service._create_translation_prompt("Japanese", "English")
        assert isinstance(system, str) and isinstance(user, str)

    def test_system_prompt_contains_languages(self, translation_service):
        system, _ = translation_service._create_translation_prompt("Japanese", "English")
        assert "Japanese" in system and "English" in system

    def test_user_prompt_contains_languages(self, translation_service):
        _, user = translation_service._create_translation_prompt("Japanese", "English")
        assert "Japanese" in user and "English" in user

    def test_file_format_changes_system_prompt(self, translation_service):
        system_console, _ = translation_service._create_translation_prompt(
            "Japanese", "English", output_format="console"
        )
        system_file, _ = translation_service._create_translation_prompt(
            "Japanese", "English", output_format="pdf"
        )
        assert system_console != system_file
