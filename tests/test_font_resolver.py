"""
Tests for font resolution utilities (no API calls, no PDF generation):
  - _fonts_dir()
  - _emit_warning()
  - get_docx_font()
  - get_pdf_font() (ImportError path only)
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

import src.output.font_resolver as fr


# ---------------------------------------------------------------------------
# _fonts_dir
# ---------------------------------------------------------------------------


class TestFontsDir:

    def test_returns_path_object(self):
        result = fr._fonts_dir()
        assert isinstance(result, Path)

    def test_ends_with_fonts_directory(self):
        result = fr._fonts_dir()
        assert result.name == "fonts"

    def test_parent_is_project_root(self):
        result = fr._fonts_dir()
        # The project root should contain main.py
        root = result.parent
        assert (root / "main.py").exists()


# ---------------------------------------------------------------------------
# _emit_warning
# ---------------------------------------------------------------------------


class TestEmitWarning:

    def test_prints_message_to_stdout(self, capsys):
        fr._emit_warning("Test warning")
        out = capsys.readouterr().out
        assert "Test warning" in out

    def test_logs_message_when_no_log_message(self, caplog):
        with caplog.at_level(logging.WARNING):
            fr._emit_warning("Warning text")
        assert "Warning text" in caplog.text

    def test_logs_custom_log_message_when_provided(self, caplog):
        with caplog.at_level(logging.WARNING):
            fr._emit_warning("Printed text", log_message="Log only text")
        assert "Log only text" in caplog.text

    def test_printed_message_is_user_facing(self, capsys):
        fr._emit_warning("User message", log_message="Internal log")
        out = capsys.readouterr().out
        assert "User message" in out
        # The internal log message should not appear in printed output
        assert "Internal log" not in out


# ---------------------------------------------------------------------------
# get_docx_font
# ---------------------------------------------------------------------------


class TestGetDocxFont:

    def test_no_custom_font_returns_arial_unicode_ms(self):
        result = fr.get_docx_font()
        assert result == "Arial Unicode MS"

    def test_none_custom_font_returns_arial_unicode_ms(self):
        result = fr.get_docx_font(None)
        assert result == "Arial Unicode MS"

    def test_custom_font_found_in_fonts_dir_returns_it(self, tmp_path):
        # Create a fake .ttf file in a temp directory mimicking fonts/
        (tmp_path / "MyFont.ttf").write_bytes(b"\x00" * 16)
        with patch.object(fr, "_fonts_dir", return_value=tmp_path):
            result = fr.get_docx_font("MyFont")
        assert result == "MyFont"

    def test_custom_font_missing_emits_warning_and_falls_back(self, tmp_path, capsys):
        # fonts dir exists but the requested font file is absent
        with patch.object(fr, "_fonts_dir", return_value=tmp_path):
            result = fr.get_docx_font("MissingFont")
        out = capsys.readouterr().out
        assert "Warning" in out
        assert result == "Arial Unicode MS"

    def test_fonts_dir_does_not_exist_falls_back_to_default(self, tmp_path):
        non_existent = tmp_path / "no_such_dir"
        with patch.object(fr, "_fonts_dir", return_value=non_existent):
            result = fr.get_docx_font("AnyFont")
        assert result == "Arial Unicode MS"


# ---------------------------------------------------------------------------
# get_pdf_font (ImportError path only — no reportlab needed)
# ---------------------------------------------------------------------------


class TestGetPdfFontImportError:

    def test_returns_times_roman_when_reportlab_unavailable(self):
        # Simulate an environment where reportlab cannot be imported
        with patch.dict("sys.modules", {"reportlab": None,
                                        "reportlab.pdfbase": None,
                                        "reportlab.pdfbase.pdfmetrics": None,
                                        "reportlab.pdfbase.ttfonts": None}):
            result = fr.get_pdf_font()
        assert result == "Times-Roman"
