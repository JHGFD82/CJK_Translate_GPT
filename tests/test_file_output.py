"""
Tests for file output utilities:
  - FileOutputHandler._normalize_paragraphs
  - FileOutputHandler._resolve_output_path
  - FileOutputHandler._emit_message
  - FileOutputHandler._ensure_parent_directory
  - FileOutputHandler._fallback_to_text
  - FileOutputHandler.save_to_text_file
  - FileOutputHandler.append_to_text_file
  - generate_output_filename
"""

import logging
from datetime import datetime
from pathlib import Path
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


# ---------------------------------------------------------------------------
# _emit_message
# ---------------------------------------------------------------------------


class TestEmitMessage:

    def test_prints_to_stdout(self, capsys):
        FileOutputHandler._emit_message("Hello world")
        out = capsys.readouterr().out
        assert "Hello world" in out

    def test_leading_newline_prepended(self, capsys):
        FileOutputHandler._emit_message("Hi", leading_newline=True)
        out = capsys.readouterr().out
        assert out.startswith("\n")

    def test_no_leading_newline_by_default(self, capsys):
        FileOutputHandler._emit_message("Hi")
        out = capsys.readouterr().out
        assert not out.startswith("\n")

    def test_logs_message(self, caplog):
        with caplog.at_level(logging.INFO):
            FileOutputHandler._emit_message("Logged message")
        assert "Logged message" in caplog.text

    def test_custom_log_message_used(self, caplog):
        with caplog.at_level(logging.INFO):
            FileOutputHandler._emit_message("Printed msg", log_message="Log msg")
        assert "Log msg" in caplog.text
        # The log should use the log_message, not the printed one
        assert "Printed msg" not in caplog.text


# ---------------------------------------------------------------------------
# _ensure_parent_directory
# ---------------------------------------------------------------------------


class TestEnsureParentDirectory:

    def test_creates_missing_parent_directory(self, tmp_path):
        deep_path = str(tmp_path / "a" / "b" / "c" / "output.txt")
        FileOutputHandler._ensure_parent_directory(deep_path)
        assert Path(deep_path).parent.exists()

    def test_existing_directory_not_an_error(self, tmp_path):
        output_path = str(tmp_path / "output.txt")
        # Parent already exists — should not raise
        FileOutputHandler._ensure_parent_directory(output_path)
        assert tmp_path.exists()


# ---------------------------------------------------------------------------
# _fallback_to_text
# ---------------------------------------------------------------------------


class TestFallbackToText:

    def test_writes_text_file_with_txt_extension(self, tmp_path):
        output_path = str(tmp_path / "output.pdf")
        FileOutputHandler._fallback_to_text("some content", output_path)
        expected = tmp_path / "output.txt"
        assert expected.exists()
        assert expected.read_text(encoding="utf-8") == "some content"

    def test_original_file_not_created(self, tmp_path):
        output_path = str(tmp_path / "output.docx")
        FileOutputHandler._fallback_to_text("content", output_path)
        assert not (tmp_path / "output.docx").exists()


# ---------------------------------------------------------------------------
# save_to_text_file
# ---------------------------------------------------------------------------


class TestSaveToTextFile:

    def test_writes_content_to_file(self, tmp_path):
        output_path = str(tmp_path / "out.txt")
        FileOutputHandler.save_to_text_file("Hello CJK: 日本語", output_path)
        assert Path(output_path).read_text(encoding="utf-8") == "Hello CJK: 日本語"

    def test_creates_file_if_not_exists(self, tmp_path):
        output_path = str(tmp_path / "new_file.txt")
        FileOutputHandler.save_to_text_file("content", output_path)
        assert Path(output_path).exists()

    def test_overwrites_existing_file(self, tmp_path):
        output_path = str(tmp_path / "out.txt")
        Path(output_path).write_text("old content", encoding="utf-8")
        FileOutputHandler.save_to_text_file("new content", output_path)
        assert Path(output_path).read_text(encoding="utf-8") == "new content"

    def test_prints_confirmation_message(self, tmp_path, capsys):
        output_path = str(tmp_path / "out.txt")
        FileOutputHandler.save_to_text_file("text", output_path)
        out = capsys.readouterr().out
        assert "saved" in out.lower() or str(output_path) in out

    def test_os_error_handled_gracefully(self, tmp_path, capsys):
        # Point at a directory so writing raises OSError
        output_path = str(tmp_path)   # directory, not a file
        # Should not raise; should print an error
        FileOutputHandler.save_to_text_file("text", output_path)
        out = capsys.readouterr().out
        assert "Error" in out or "error" in out


# ---------------------------------------------------------------------------
# append_to_text_file
# ---------------------------------------------------------------------------


class TestAppendToTextFile:

    def test_appends_content_to_existing_file(self, tmp_path):
        output_path = str(tmp_path / "out.txt")
        Path(output_path).write_text("Page 1", encoding="utf-8")
        FileOutputHandler.append_to_text_file("Page 2", output_path)
        content = Path(output_path).read_text(encoding="utf-8")
        assert "Page 1" in content
        assert "Page 2" in content

    def test_creates_file_if_not_exists(self, tmp_path):
        output_path = str(tmp_path / "out.txt")
        FileOutputHandler.append_to_text_file("content", output_path)
        assert Path(output_path).exists()

    def test_appended_content_followed_by_double_newline(self, tmp_path):
        output_path = str(tmp_path / "out.txt")
        FileOutputHandler.append_to_text_file("chunk", output_path)
        content = Path(output_path).read_text(encoding="utf-8")
        assert content.endswith("\n\n")

    def test_multiple_appends_accumulate(self, tmp_path):
        output_path = str(tmp_path / "out.txt")
        for i in range(3):
            FileOutputHandler.append_to_text_file(f"chunk {i}", output_path)
        content = Path(output_path).read_text(encoding="utf-8")
        for i in range(3):
            assert f"chunk {i}" in content

    def test_prints_confirmation(self, tmp_path, capsys):
        output_path = str(tmp_path / "out.txt")
        FileOutputHandler.append_to_text_file("text", output_path)
        out = capsys.readouterr().out
        assert str(output_path) in out or "appended" in out.lower() or "Page" in out
