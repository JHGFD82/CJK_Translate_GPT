"""Tests for the prompt spec classes in src/services/prompts/.

Focus: spec interface, note injection, dataclass defaults, and interactions
between parameters that are not exercised by the service-level tests in
test_prompts.py.
"""

import pytest

from src.services.prompts import OcrPromptSpec, ImageTranslationPromptSpec, TranslationPromptSpec


# ===========================================================================
# TranslationPromptSpec
# ===========================================================================

class TestTranslationPromptSpecDefaults:
    """Ensure spec works out of the box with only the required fields."""

    def test_returns_strings(self):
        spec = TranslationPromptSpec("Chinese", "English")
        assert isinstance(spec.system_prompt(), str)
        assert isinstance(spec.user_prompt(), str)

    def test_default_output_format_is_console(self):
        spec = TranslationPromptSpec("Chinese", "English")
        assert "console" in spec.system_prompt().lower()

    def test_default_has_no_numbered_block(self):
        spec = TranslationPromptSpec("Chinese", "English")
        assert "numbered lists" not in spec.system_prompt()
        assert "CRITICAL" not in spec.user_prompt()

    def test_user_prompt_ends_with_double_newline(self):
        # The caller concatenates source text directly onto user_prompt().
        spec = TranslationPromptSpec("Chinese", "English")
        assert spec.user_prompt().endswith("\n\n")


class TestTranslationPromptSpecNoteInjection:
    """system_note and user_note are injected into the correct prompt."""

    def test_system_note_appears_in_system_prompt(self):
        spec = TranslationPromptSpec("Chinese", "English", system_note="MYSYSNOTE")
        assert "MYSYSNOTE" in spec.system_prompt()
        assert "MYSYSNOTE" not in spec.user_prompt()

    def test_user_note_appears_in_user_prompt(self):
        spec = TranslationPromptSpec("Chinese", "English", user_note="MYUSERNOTE")
        assert "MYUSERNOTE" in spec.user_prompt()
        assert "MYUSERNOTE" not in spec.system_prompt()

    def test_system_note_label_present(self):
        spec = TranslationPromptSpec("Chinese", "English", system_note="xyz")
        assert "ADDITIONAL INSTRUCTIONS" in spec.system_prompt()

    def test_user_note_label_present(self):
        spec = TranslationPromptSpec("Chinese", "English", user_note="xyz")
        assert "ADDITIONAL NOTES" in spec.user_prompt()

    def test_no_note_labels_when_notes_absent(self):
        spec = TranslationPromptSpec("Chinese", "English")
        assert "ADDITIONAL INSTRUCTIONS" not in spec.system_prompt()
        assert "ADDITIONAL NOTES" not in spec.user_prompt()


class TestTranslationPromptSpecOutputFormat:
    """output_format drives the formatting fragment in the system prompt."""

    @pytest.mark.parametrize("fmt", ["pdf", "txt", "docx", "file", "PDF", "DOCX"])
    def test_file_formats_use_file_fragment(self, fmt):
        spec = TranslationPromptSpec("Chinese", "English", output_format=fmt)
        assert "file output" in spec.system_prompt()
        assert "console" not in spec.system_prompt().lower()

    @pytest.mark.parametrize("fmt", ["console", "", "screen", "unknown"])
    def test_non_file_formats_use_console_fragment(self, fmt):
        spec = TranslationPromptSpec("Chinese", "English", output_format=fmt)
        assert "console" in spec.system_prompt().lower()
        assert "file output" not in spec.system_prompt()


class TestTranslationPromptSpecNumbered:
    """has_numbered gates two independent fragments (one per prompt)."""

    def test_system_prompt_gets_numbered_block(self):
        spec = TranslationPromptSpec("Japanese", "English", has_numbered=True)
        assert "numbered lists" in spec.system_prompt()

    def test_user_prompt_gets_continuation_block(self):
        spec = TranslationPromptSpec("Japanese", "English", has_numbered=True)
        assert "NUMBERING CONTINUATION" in spec.user_prompt()

    def test_both_blocks_absent_when_false(self):
        spec = TranslationPromptSpec("Japanese", "English", has_numbered=False)
        assert "numbered lists" not in spec.system_prompt()
        assert "NUMBERING CONTINUATION" not in spec.user_prompt()


class TestTranslationPromptSpecLanguagePairNotes:
    """Automatic honorific/formality notes for JK and KJ pairs."""

    def test_jk_pair_injects_honorific_note(self):
        spec = TranslationPromptSpec("Japanese", "Korean")
        assert "honorific" in spec.system_prompt().lower()

    def test_kj_pair_injects_honorific_note(self):
        spec = TranslationPromptSpec("Korean", "Japanese")
        assert "honorific" in spec.system_prompt().lower()

    def test_pair_note_not_present_for_unrelated_pair(self):
        spec = TranslationPromptSpec("Chinese", "English")
        # No pair note — ADDITIONAL INSTRUCTIONS block should be absent
        # (unless a system_note was explicitly passed)
        assert "honorific" not in spec.system_prompt()

    def test_explicit_system_note_appended_after_pair_note(self):
        spec = TranslationPromptSpec("Japanese", "Korean", system_note="EXPLICIT")
        prompt = spec.system_prompt()
        honorific_pos = prompt.lower().index("honorific")
        explicit_pos = prompt.index("EXPLICIT")
        assert explicit_pos > honorific_pos  # pair note comes first

    def test_kj_note_mentions_teineigo(self):
        spec = TranslationPromptSpec("Korean", "Japanese")
        assert "teineigo" in spec.system_prompt().lower() or "丁寧語" in spec.system_prompt()

    def test_jk_note_mentions_korean_formal_register(self):
        spec = TranslationPromptSpec("Japanese", "Korean")
        assert "합쇼체" in spec.system_prompt() or "해요체" in spec.system_prompt()


class TestTranslationPromptSpecKanbun:
    """--kanbun flag injects kundoku reading conventions."""

    def test_kanbun_false_by_default(self):
        spec = TranslationPromptSpec("Chinese", "English")
        prompt = spec.system_prompt()
        assert "kundoku" not in prompt.lower()

    def test_kanbun_true_injects_note(self):
        spec = TranslationPromptSpec("Chinese", "English", kanbun=True)
        prompt = spec.system_prompt()
        assert "kundoku" in prompt.lower() or "kanbun" in prompt.lower() or "漢文" in prompt

    def test_kanbun_note_absent_when_false(self):
        spec = TranslationPromptSpec("Chinese", "English", kanbun=False)
        from src.services.prompts import fragments as F
        assert F.KANBUN_NOTE not in spec.system_prompt()

    def test_kanbun_true_injects_kanbun_note_constant(self):
        spec = TranslationPromptSpec("Chinese", "English", kanbun=True)
        from src.services.prompts import fragments as F
        assert F.KANBUN_NOTE in spec.system_prompt()

    def test_kanbun_note_appears_after_pair_note(self):
        # JK pair note + kanbun — unusual but possible; kanbun note should follow pair note
        spec = TranslationPromptSpec("Japanese", "Korean", kanbun=True)
        prompt = spec.system_prompt()
        pair_pos = prompt.lower().index("honorific")
        kanbun_pos = prompt.index("漢文") if "漢文" in prompt else prompt.lower().index("kanbun")
        assert kanbun_pos > pair_pos

    def test_kanbun_and_system_note_both_present(self):
        spec = TranslationPromptSpec("Chinese", "English", kanbun=True, system_note="MYSYSNOTE")
        prompt = spec.system_prompt()
        assert "MYSYSNOTE" in prompt
        assert "漢文" in prompt or "kanbun" in prompt.lower() or "kundoku" in prompt.lower()

    def test_kanbun_system_note_ordering(self):
        spec = TranslationPromptSpec("Chinese", "English", kanbun=True, system_note="AFTERKANBUN")
        prompt = spec.system_prompt()
        from src.services.prompts import fragments as F
        kanbun_pos = prompt.index(F.KANBUN_NOTE)
        sysnote_pos = prompt.index("AFTERKANBUN")
        assert sysnote_pos > kanbun_pos


# ===========================================================================
# OcrPromptSpec
# ===========================================================================

class TestOcrPromptSpecDefaults:
    """Ensure spec works out of the box with only the required field."""

    def test_returns_strings(self):
        spec = OcrPromptSpec("English")
        assert isinstance(spec.system_prompt(), str)
        assert isinstance(spec.user_prompt(), str)
        assert isinstance(spec.refinement_prompt(), str)

    def test_default_is_not_vertical(self):
        spec = OcrPromptSpec("English")
        assert "TEXT ORIENTATION" not in spec.system_prompt()
        assert "ORIENTATION REMINDER" not in spec.user_prompt()
        assert "ORIENTATION REMINDER" not in spec.refinement_prompt()


class TestOcrPromptSpecNoteInjection:
    """Notes are injected only into the expected prompt half."""

    def test_system_note_in_system_prompt_only(self):
        spec = OcrPromptSpec("Japanese", system_note="SYSCHECK")
        assert "SYSCHECK" in spec.system_prompt()
        assert "SYSCHECK" not in spec.user_prompt()

    def test_user_note_in_user_prompt_only(self):
        spec = OcrPromptSpec("Japanese", user_note="USERCHECK")
        assert "USERCHECK" in spec.user_prompt()
        assert "USERCHECK" not in spec.system_prompt()

    def test_refinement_prompt_omits_notes_by_design(self):
        # Refinement prompts are note-agnostic (they review a prior transcript).
        spec = OcrPromptSpec("Japanese", system_note="SYS", user_note="USR")
        assert "SYS" not in spec.refinement_prompt()
        assert "USR" not in spec.refinement_prompt()


class TestOcrPromptSpecVertical:
    """Vertical flag adds orientation blocks in system, user and refinement prompts."""

    def test_system_prompt_gets_orientation_block(self):
        spec = OcrPromptSpec("Japanese", vertical=True)
        assert "TEXT ORIENTATION" in spec.system_prompt()

    def test_user_prompt_gets_orientation_reminder(self):
        spec = OcrPromptSpec("Japanese", vertical=True)
        assert "ORIENTATION REMINDER" in spec.user_prompt()

    def test_refinement_prompt_gets_orientation_reminder(self):
        spec = OcrPromptSpec("Japanese", vertical=True)
        assert "ORIENTATION REMINDER" in spec.refinement_prompt()


class TestOcrPromptSpecScriptGuidance:
    """Script guidance is language-keyed and appears in all three prompts."""

    @pytest.mark.parametrize("lang,keyword", [
        ("Japanese", "hiragana"),
        ("Korean", "hangul"),
        ("Simplified Chinese", "简体字"),
        ("Traditional Chinese", "繁體字"),
        ("English", "Latin alphabet"),
    ])
    def test_script_guidance_in_system_prompt(self, lang, keyword):
        assert keyword in OcrPromptSpec(lang).system_prompt()

    @pytest.mark.parametrize("lang,keyword", [
        ("Japanese", "hiragana"),
        ("Korean", "hangul"),
        ("English", "Latin alphabet"),
    ])
    def test_script_guidance_in_user_prompt(self, lang, keyword):
        assert keyword in OcrPromptSpec(lang).user_prompt()

    @pytest.mark.parametrize("lang,keyword", [
        ("Japanese", "hiragana"),
        ("Korean", "hangul"),
        ("English", "Latin alphabet"),
    ])
    def test_script_guidance_in_refinement_prompt(self, lang, keyword):
        assert keyword in OcrPromptSpec(lang).refinement_prompt()

    def test_unknown_language_has_no_script_section(self):
        spec = OcrPromptSpec("Spanish")
        assert "SCRIPT NOTES:" not in spec.system_prompt()
        assert "SCRIPT REMINDER:" not in spec.user_prompt()


# ===========================================================================
# ImageTranslationPromptSpec
# ===========================================================================

class TestImageTranslationPromptSpecDefaults:
    """Ensure spec works out of the box with the required fields."""

    def test_returns_strings(self):
        spec = ImageTranslationPromptSpec("Japanese", "English")
        assert isinstance(spec.system_prompt(), str)
        assert isinstance(spec.user_prompt(), str)

    def test_always_contains_response_format_headers(self):
        spec = ImageTranslationPromptSpec("Japanese", "English")
        assert "[TRANSCRIPT]" in spec.system_prompt()
        assert "[TRANSLATION]" in spec.system_prompt()

    def test_default_is_not_vertical(self):
        spec = ImageTranslationPromptSpec("Japanese", "English")
        assert "vertical" not in spec.user_prompt()
        assert "TEXT ORIENTATION" not in spec.system_prompt()


class TestImageTranslationPromptSpecNoteInjection:
    """Notes routed to the correct prompt half."""

    def test_system_note_in_system_prompt_only(self):
        spec = ImageTranslationPromptSpec("Japanese", "English", system_note="SYSCHECK")
        assert "SYSCHECK" in spec.system_prompt()
        assert "SYSCHECK" not in spec.user_prompt()

    def test_user_note_in_user_prompt_only(self):
        spec = ImageTranslationPromptSpec("Japanese", "English", user_note="USERCHECK")
        assert "USERCHECK" in spec.user_prompt()
        assert "USERCHECK" not in spec.system_prompt()


class TestImageTranslationPromptSpecVertical:
    """vertical=True adds orientation blocks in both prompts."""

    def test_system_prompt_gets_orientation_block(self):
        spec = ImageTranslationPromptSpec("Japanese", "English", vertical=True)
        assert "TEXT ORIENTATION" in spec.system_prompt()

    def test_user_prompt_gets_vertical_note(self):
        spec = ImageTranslationPromptSpec("Japanese", "English", vertical=True)
        assert "vertical" in spec.user_prompt()

    def test_no_orientation_without_flag(self):
        spec = ImageTranslationPromptSpec("Japanese", "English", vertical=False)
        assert "TEXT ORIENTATION" not in spec.system_prompt()
        assert "vertical" not in spec.user_prompt()


class TestImageTranslationPromptSpecScriptGuidance:
    """Script guidance keyed by source language (not target)."""

    def test_japanese_source_guidance_present(self):
        spec = ImageTranslationPromptSpec("Japanese", "English")
        assert "SCRIPT NOTES:" in spec.system_prompt()
        assert "kanji" in spec.system_prompt()

    def test_unknown_source_language_no_guidance(self):
        spec = ImageTranslationPromptSpec("Spanish", "English")
        assert "SCRIPT NOTES:" not in spec.system_prompt()

    def test_guidance_keyed_by_source_not_target(self):
        # Japanese source should appear regardless of target language
        spec_a = ImageTranslationPromptSpec("Japanese", "Korean")
        spec_b = ImageTranslationPromptSpec("Korean", "Japanese")
        assert "kanji" in spec_a.system_prompt()
        assert "kanji" not in spec_b.system_prompt()


# ===========================================================================
# Cross-spec invariants
# ===========================================================================

class TestSharedBehaviours:
    """Properties that should hold across all three spec types."""

    def test_all_prompts_are_non_empty(self):
        specs = [
            TranslationPromptSpec("Chinese", "English"),
            OcrPromptSpec("Japanese"),
            ImageTranslationPromptSpec("Japanese", "English"),
        ]
        for spec in specs:
            assert spec.system_prompt().strip()
            assert spec.user_prompt().strip()

    def test_note_injection_is_additive(self):
        """Adding a note should only extend, never shorten, the prompt."""
        base = TranslationPromptSpec("Chinese", "English")
        noted = TranslationPromptSpec("Chinese", "English", system_note="extra")
        assert len(noted.system_prompt()) > len(base.system_prompt())
        assert len(noted.user_prompt()) == len(base.user_prompt())
