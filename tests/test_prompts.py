"""
Tests for prompt construction across all three services:
  - ImageProcessorService  (_build_system_prompt, _build_user_prompt, _create_ocr_prompt)
  - ImageTranslationService (_build_system_prompt, _build_user_prompt)
  - TranslationService      (_get_formatting_instruction, _build_system_prompt,
                             _build_user_prompt_template, _create_translation_prompt)

No API calls are made; all methods under test are pure string builders.
"""

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains(text: str, *fragments: str) -> bool:
    """Return True only if every fragment appears in text."""
    return all(f in text for f in fragments)


# ===========================================================================
# ImageProcessorService prompt tests
# ===========================================================================

class TestOCRSystemPrompt:
    """_build_system_prompt for the standalone OCR service."""

    def test_always_contains_rules_section(self, ocr_service):
        result = ocr_service._build_system_prompt("Japanese")
        assert "RULES:" in result

    def test_always_contains_do_not_translate(self, ocr_service):
        result = ocr_service._build_system_prompt("Japanese")
        assert "Do NOT translate" in result or "do NOT translate" in result.lower()

    def test_no_vertical_section_by_default(self, ocr_service):
        result = ocr_service._build_system_prompt("Japanese")
        assert "TEXT ORIENTATION" not in result
        assert "top-to-bottom" not in result
        assert "right-to-left" not in result

    def test_vertical_section_present_when_flag_set(self, ocr_service):
        result = ocr_service._build_system_prompt("Japanese", vertical=True)
        assert "TEXT ORIENTATION" in result
        assert "top-to-bottom" in result
        assert "right-to-left" in result

    def test_vertical_describes_column_order(self, ocr_service):
        result = ocr_service._build_system_prompt("Japanese", vertical=True)
        assert "rightmost column" in result

    def test_japanese_script_guidance_injected(self, ocr_service):
        result = ocr_service._build_system_prompt("Japanese")
        assert "SCRIPT NOTES:" in result
        assert "Japanese script" in result
        assert "hiragana" in result

    def test_english_script_guidance_injected(self, ocr_service):
        result = ocr_service._build_system_prompt("English")
        assert "SCRIPT NOTES:" in result
        assert "Latin alphabet" in result

    def test_korean_script_guidance_injected(self, ocr_service):
        result = ocr_service._build_system_prompt("Korean")
        assert "SCRIPT NOTES:" in result
        assert "hangul" in result

    def test_simplified_chinese_script_guidance(self, ocr_service):
        result = ocr_service._build_system_prompt("Simplified Chinese")
        assert "Simplified Chinese" in result
        assert "简体字" in result

    def test_traditional_chinese_script_guidance(self, ocr_service):
        result = ocr_service._build_system_prompt("Traditional Chinese")
        assert "Traditional Chinese" in result
        assert "繁體字" in result

    def test_unknown_language_has_no_script_notes(self, ocr_service):
        result = ocr_service._build_system_prompt("Spanish")
        assert "SCRIPT NOTES:" not in result

    def test_vertical_and_script_notes_both_present(self, ocr_service):
        result = ocr_service._build_system_prompt("Japanese", vertical=True)
        assert "SCRIPT NOTES:" in result
        assert "TEXT ORIENTATION" in result


class TestOCRUserPrompt:
    """_build_user_prompt for the standalone OCR service."""

    def test_contains_target_language(self, ocr_service):
        result = ocr_service._build_user_prompt("Japanese")
        assert "Japanese" in result

    def test_contains_do_not_translate(self, ocr_service):
        result = ocr_service._build_user_prompt("Japanese")
        assert "Do NOT translate" in result

    def test_no_vertical_note_by_default(self, ocr_service):
        result = ocr_service._build_user_prompt("Japanese")
        assert "vertical" not in result

    def test_vertical_note_added_when_flag_set(self, ocr_service):
        result = ocr_service._build_user_prompt("Japanese", vertical=True)
        assert "vertical" in result
        assert "top-to-bottom" in result
        assert "right-to-left" in result

    def test_different_language_reflected(self, ocr_service):
        result = ocr_service._build_user_prompt("Korean")
        assert "Korean" in result
        assert "Japanese" not in result


class TestCreateOCRPrompt:
    """_create_ocr_prompt delegates to the two builders; verify tuple structure."""

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
# ImageTranslationService prompt tests
# ===========================================================================

class TestImageTranslationSystemPrompt:
    """_build_system_prompt for the combined OCR + translation service."""

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
# TranslationService prompt tests
# ===========================================================================

class TestFormattingInstruction:
    """_get_formatting_instruction — file vs. console mode."""

    @pytest.mark.parametrize("fmt", ["pdf", "PDF", "txt", "TXT", "docx", "DOCX", "file", "FILE"])
    def test_file_formats_return_file_instruction(self, translation_service, fmt):
        result = translation_service._get_formatting_instruction(fmt)
        assert "file output" in result
        assert "actual line breaks" in result

    @pytest.mark.parametrize("fmt", ["console", "CONSOLE", "", "unknown", "screen"])
    def test_non_file_formats_return_console_instruction(self, translation_service, fmt):
        result = translation_service._get_formatting_instruction(fmt)
        assert "console" in result.lower()


class TestTranslationSystemPrompt:
    """_build_system_prompt for the text translation service."""

    def test_contains_source_language(self, translation_service):
        result = translation_service._build_system_prompt(
            "Japanese", "English", "file instruction", "numbering instruction"
        )
        assert "Japanese" in result

    def test_contains_target_language(self, translation_service):
        result = translation_service._build_system_prompt(
            "Japanese", "English", "file instruction", "numbering instruction"
        )
        assert "English" in result

    def test_formatting_instruction_embedded(self, translation_service):
        sentinel = "SENTINEL_FORMATTING_TEXT"
        result = translation_service._build_system_prompt(
            "Japanese", "English", sentinel, "some numbering"
        )
        assert sentinel in result

    def test_numbered_content_instruction_embedded(self, translation_service):
        sentinel = "SENTINEL_NUMBERING_TEXT"
        result = translation_service._build_system_prompt(
            "Japanese", "English", "some formatting", sentinel
        )
        assert sentinel in result

    def test_mentions_translation_role(self, translation_service):
        result = translation_service._build_system_prompt(
            "Japanese", "English", "fmt", "num"
        )
        assert "professional translator" in result


class TestTranslationUserPromptTemplate:
    """_build_user_prompt_template for the text translation service."""

    def test_contains_source_language(self, translation_service):
        result = translation_service._build_user_prompt_template("Japanese", "English")
        assert "Japanese" in result

    def test_contains_target_language(self, translation_service):
        result = translation_service._build_user_prompt_template("Japanese", "English")
        assert "English" in result

    def test_references_current_page_marker(self, translation_service):
        result = translation_service._build_user_prompt_template("Japanese", "English")
        assert "--Current Page:" in result

    def test_mentions_numbering_preservation(self, translation_service):
        # has_numbered=True → numbering sections are included
        result = translation_service._build_user_prompt_template("Japanese", "English", has_numbered=True)
        assert "numbering" in result.lower() or "CRITICAL" in result

    def test_no_numbering_instructions_when_not_detected(self, translation_service):
        # has_numbered=False (default) → numbering sections are omitted
        result = translation_service._build_user_prompt_template("Japanese", "English", has_numbered=False)
        assert "CRITICAL" not in result
        assert "NUMBERING CONTINUATION" not in result


class TestCreateTranslationPrompt:
    """_create_translation_prompt assembles both prompts; verify tuple structure."""

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
