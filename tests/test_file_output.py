"""
Tests for file output utilities:
  - FileOutputHandler._normalize_paragraphs
  - FileOutputHandler._resolve_output_path
  - generate_output_filename
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.output.file_output import FileOutputHandler, generate_output_filename


# ---------------------------------------------------------------------------
# _normalize_paragraphs
# ---------------------------------------------------------------------------

class TestNormalizeParagraphs:

    def test_empty_string_returns_empty_list(self):
        assert FileOutputHandler._normalize_paragraphs("") == []

    def test_single_paragraph(self):
        assert FileOutputHandler._normalize_paragraphs("Hello World") == ["Hello World"]

    def test_double_newline_splits_paragraphs(self):
        result = FileOutputHandler._normalize_paragraphs("Hello\n\nWorld")
        assert result == ["Hello", "World"]

    def test_empty_paragraphs_filtered_out(self):
        # Extra blank lines produce empty paragraphs, which are discarded
        result = FileOutputHandler._normalize_paragraphs("Hello\n\n\n\nWorld")
        assert result == ["Hello", "World"]

    def test_inner_newlines_replaced_with_space(self):
        # Within a double-newline block, single newlines become spaces
        result = FileOutputHandler._normalize_paragraphs("Line1\nLine2\n\nLine3")
        assert result == ["Line1 Line2", "Line3"]

    def test_strips_leading_trailing_whitespace_from_paragraphs(self):
        result = FileOutputHandler._normalize_paragraphs("  Hello  \n\n  World  ")
        assert result == ["Hello", "World"]

    def test_multiple_paragraphs(self):
        content = "One\n\nTwo\n\nThree"
        assert FileOutputHandler._normalize_paragraphs(content) == ["One", "Two", "Three"]

    def test_whitespace_only_paragraph_filtered(self):
        result = FileOutputHandler._normalize_paragraphs("Real\n\n   \n\nContent")
        assert result == ["Real", "Content"]

    def test_cjk_content_preserved(self):
        result = FileOutputHandler._normalize_paragraphs("日本語\n\n中文")
        assert result == ["日本語", "中文"]


# ---------------------------------------------------------------------------
# _resolve_output_path
# ---------------------------------------------------------------------------

class TestResolveOutputPath:

    def test_explicit_output_file_returned_directly(self):
        result = FileOutputHandler._resolve_output_path(
            input_file="/input/doc.pdf",
            output_file="/output/result.txt",
            auto_save=False,
            source_lang="Japanese",
            target_lang="English",
        )
        assert result == "/output/result.txt"

    def test_explicit_output_file_takes_priority_over_auto_save(self):
        result = FileOutputHandler._resolve_output_path(
            input_file="/input/doc.pdf",
            output_file="/output/result.txt",
            auto_save=True,
            source_lang="Japanese",
            target_lang="English",
        )
        assert result == "/output/result.txt"

    def test_auto_save_generates_timestamped_name(self):
        result = FileOutputHandler._resolve_output_path(
            input_file="/input/doc.pdf",
            output_file=None,
            auto_save=True,
            source_lang="Japanese",
            target_lang="English",
        )
        assert result is not None
        assert "doc_JapanesetoEnglish_" in result
        assert result.endswith(".txt")

    def test_auto_save_output_placed_beside_input(self):
        result = FileOutputHandler._resolve_output_path(
            input_file="/some/folder/doc.pdf",
            output_file=None,
            auto_save=True,
            source_lang="Japanese",
            target_lang="English",
        )
        assert result is not None
        assert result.startswith("/some/folder/")

    def test_no_output_no_auto_save_returns_none(self):
        result = FileOutputHandler._resolve_output_path(
            input_file="/input/doc.pdf",
            output_file=None,
            auto_save=False,
            source_lang="Japanese",
            target_lang="English",
        )
        assert result is None

    def test_auto_save_without_input_file_returns_none(self):
        result = FileOutputHandler._resolve_output_path(
            input_file=None,
            output_file=None,
            auto_save=True,
            source_lang="Japanese",
            target_lang="English",
        )
        assert result is None

    def test_custom_extension_honoured(self):
        result = FileOutputHandler._resolve_output_path(
            input_file="/input/doc.pdf",
            output_file=None,
            auto_save=True,
            source_lang="Japanese",
            target_lang="English",
            default_extension=".docx",
        )
        assert result is not None
        assert result.endswith(".docx")


# ---------------------------------------------------------------------------
# generate_output_filename
# ---------------------------------------------------------------------------

class TestGenerateOutputFilename:

    def test_filename_contains_source_and_target_language(self):
        result = generate_output_filename("/input/doc.pdf", "Japanese", "English")
        assert "JapanesetoEnglish" in result

    def test_filename_uses_input_stem(self):
        result = generate_output_filename("/input/my_document.pdf", "Japanese", "English")
        assert "my_document_JapanesetoEnglish_" in result

    def test_output_placed_in_same_directory_as_input(self):
        result = generate_output_filename("/some/path/doc.pdf", "Japanese", "English")
        assert result.startswith("/some/path/")

    def test_default_extension_is_txt(self):
        result = generate_output_filename("/input/doc.pdf", "Japanese", "English")
        assert result.endswith(".txt")

    def test_custom_extension_applied(self):
        result = generate_output_filename("/input/doc.pdf", "Japanese", "English", ".docx")
        assert result.endswith(".docx")

    def test_timestamp_format_in_filename(self):
        with patch("src.output.file_output.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20260311_143000"
            result = generate_output_filename("/input/doc.pdf", "Japanese", "English")
        assert "20260311_143000" in result

    def test_different_languages_produce_different_filenames(self):
        r1 = generate_output_filename("/input/doc.pdf", "Japanese", "English")
        r2 = generate_output_filename("/input/doc.pdf", "Chinese", "English")
        # Strip the timestamp portion to compare just the language tags
        assert "JapanesetoEnglish" in r1
        assert "ChinesetoEnglish" in r2
