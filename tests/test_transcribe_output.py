"""
Regression tests for transcribe command output routing.

Verifies that process_image() and process_image_folder() route their output
through FileOutputHandler so that .pdf → save_to_pdf, .docx → save_to_docx,
and .txt → save_to_text_file are called — never a raw text write into a
mismatched extension (the original corruption bug).

No API calls or real image files are needed; the OCR service and all I/O
helpers are mocked.
"""

import os
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.runtime.sandbox_processor import SandboxProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OCR_TEXT = "Line one.\n\nLine two."
OCR_TEXT_IMG2 = "Line three.\n\nLine four."


def _make_tiny_png() -> bytes:
    """Return bytes of a 1×1 red PNG (no external dependencies)."""
    sig = b'\x89PNG\r\n\x1a\n'

    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = struct.pack('>I', len(data)) + ctype + data
        return c + struct.pack('>I', zlib.crc32(ctype + data) & 0xFFFFFFFF)

    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
    raw = b'\x00\xff\x00\x00'
    idat = chunk(b'IDAT', zlib.compress(raw))
    iend = chunk(b'IEND', b'')
    return sig + ihdr + idat + iend


@pytest.fixture
def processor():
    """SandboxProcessor with all external calls mocked out."""
    with patch("src.runtime.sandbox_processor.get_api_key", return_value=("fake-key", "Test Prof")), \
         patch("src.runtime.sandbox_processor.TokenTracker"), \
         patch("src.runtime.sandbox_processor.TranslationService"), \
         patch("src.runtime.sandbox_processor.ImageProcessorService") as mock_ocr_cls, \
         patch("src.runtime.sandbox_processor.ImageTranslationService"), \
         patch("src.runtime.sandbox_processor.PromptService"), \
         patch("src.runtime.sandbox_processor.ImageProcessor"), \
         patch("src.runtime.sandbox_processor.PDFProcessor"):

        sp = SandboxProcessor("testprof")

        # Make the OCR service return predictable text
        sp.image_processor_service.process_image_ocr.return_value = OCR_TEXT

        yield sp


# ---------------------------------------------------------------------------
# process_image — single image
# ---------------------------------------------------------------------------

class TestProcessImageOutputRouting:

    def test_txt_output_calls_save_to_text_file(self, processor, tmp_path):
        img = tmp_path / "scan.png"
        img.write_bytes(_make_tiny_png())
        out = str(tmp_path / "result.txt")

        with patch.object(processor.file_output, "save_translation_output") as mock_save:
            processor.process_image(str(img), "English", output_file=out)

        mock_save.assert_called_once()
        _, call_kwargs = mock_save.call_args[0], mock_save.call_args
        # First positional arg is the text content
        assert mock_save.call_args[0][0] == OCR_TEXT
        # Third positional arg is the output_file path
        assert mock_save.call_args[0][2] == out

    def test_pdf_output_routes_to_save_translation_output(self, processor, tmp_path):
        """A .pdf output path must NOT produce a corrupt plain-text file."""
        img = tmp_path / "scan.png"
        img.write_bytes(_make_tiny_png())
        out = str(tmp_path / "result.pdf")

        with patch.object(processor.file_output, "save_translation_output") as mock_save, \
             patch("src.output.file_output.FileOutputHandler.save_to_pdf") as mock_pdf:
            # Wire save_translation_output through to the real body so we can
            # confirm save_to_pdf is called rather than a raw open() write
            processor.file_output.save_translation_output.side_effect = None
            from src.output.file_output import FileOutputHandler
            processor.file_output.save_translation_output = FileOutputHandler.save_translation_output

            processor.process_image(str(img), "English", output_file=out)

        mock_pdf.assert_called_once()
        # Confirm content is the OCR text, not a corrupt file
        assert mock_pdf.call_args[0][0] == OCR_TEXT
        assert mock_pdf.call_args[0][1] == out

    def test_docx_output_routes_to_save_to_docx(self, processor, tmp_path):
        img = tmp_path / "scan.png"
        img.write_bytes(_make_tiny_png())
        out = str(tmp_path / "result.docx")

        with patch("src.output.file_output.FileOutputHandler.save_to_docx") as mock_docx:
            from src.output.file_output import FileOutputHandler
            processor.file_output.save_translation_output = FileOutputHandler.save_translation_output

            processor.process_image(str(img), "English", output_file=out)

        mock_docx.assert_called_once()
        assert mock_docx.call_args[0][0] == OCR_TEXT
        assert mock_docx.call_args[0][1] == out

    def test_no_output_file_does_not_save(self, processor, tmp_path):
        img = tmp_path / "scan.png"
        img.write_bytes(_make_tiny_png())

        with patch.object(processor.file_output, "save_translation_output") as mock_save:
            processor.process_image(str(img), "English", output_file=None)

        mock_save.assert_not_called()

    def test_txt_output_is_valid_utf8_text(self, processor, tmp_path):
        """End-to-end: the .txt file must contain the OCR text, not binary."""
        img = tmp_path / "scan.png"
        img.write_bytes(_make_tiny_png())
        out = tmp_path / "result.txt"

        from src.output.file_output import FileOutputHandler
        processor.file_output.save_translation_output = FileOutputHandler.save_translation_output

        processor.process_image(str(img), "English", output_file=str(out))

        content = out.read_text(encoding="utf-8")
        assert content == OCR_TEXT

    def test_pdf_output_produces_valid_pdf_header(self, processor, tmp_path):
        """End-to-end: the .pdf file must start with %%PDF-, not plain text."""
        img = tmp_path / "scan.png"
        img.write_bytes(_make_tiny_png())
        out = tmp_path / "result.pdf"

        from src.output.file_output import FileOutputHandler
        processor.file_output.save_translation_output = FileOutputHandler.save_translation_output

        processor.process_image(str(img), "English", output_file=str(out))

        assert out.exists(), "PDF output file was not created"
        header = out.read_bytes()[:5]
        assert header == b"%PDF-", f"Expected PDF header, got {header!r}"


# ---------------------------------------------------------------------------
# process_image_folder — folder of images
# ---------------------------------------------------------------------------

class TestProcessImageFolderOutputRouting:

    def _make_folder(self, tmp_path: Path) -> Path:
        folder = tmp_path / "scans"
        folder.mkdir()
        for name in ("a.png", "b.png"):
            (folder / name).write_bytes(_make_tiny_png())
        return folder

    def test_txt_output_combined_text_written(self, processor, tmp_path):
        folder = self._make_folder(tmp_path)
        out = str(tmp_path / "combined.txt")

        from src.output.file_output import FileOutputHandler
        processor.file_output.save_translation_output = FileOutputHandler.save_translation_output

        processor.process_image_folder(str(folder), "English", output_file=out)

        content = Path(out).read_text(encoding="utf-8")
        assert "a.png" in content
        assert "b.png" in content
        assert OCR_TEXT in content

    def test_pdf_output_produces_valid_pdf_header(self, processor, tmp_path):
        """Regression: folder OCR with -o result.pdf must not produce a corrupt file."""
        folder = self._make_folder(tmp_path)
        out = tmp_path / "combined.pdf"

        from src.output.file_output import FileOutputHandler
        processor.file_output.save_translation_output = FileOutputHandler.save_translation_output

        processor.process_image_folder(str(folder), "English", output_file=str(out))

        assert out.exists(), "PDF output file was not created"
        header = out.read_bytes()[:5]
        assert header == b"%PDF-", f"Expected PDF header, got {header!r}"

    def test_pdf_output_does_not_start_with_plain_text(self, processor, tmp_path):
        """The output must not be raw UTF-8 text masquerading as a PDF."""
        folder = self._make_folder(tmp_path)
        out = tmp_path / "combined.pdf"

        from src.output.file_output import FileOutputHandler
        processor.file_output.save_translation_output = FileOutputHandler.save_translation_output

        processor.process_image_folder(str(folder), "English", output_file=str(out))

        raw = out.read_bytes()
        assert not raw.startswith(b"=== "), \
            "Output starts with plain text section header — file is corrupt"

    def test_docx_output_produces_valid_docx(self, processor, tmp_path):
        """End-to-end: .docx output must be a valid ZIP (OOXML) file."""
        folder = self._make_folder(tmp_path)
        out = tmp_path / "combined.docx"

        from src.output.file_output import FileOutputHandler
        processor.file_output.save_translation_output = FileOutputHandler.save_translation_output

        processor.process_image_folder(str(folder), "English", output_file=str(out))

        assert out.exists(), "DOCX output file was not created"
        # DOCX files are ZIP archives; they start with the ZIP magic bytes
        header = out.read_bytes()[:2]
        assert header == b"PK", f"Expected ZIP/DOCX header PK, got {header!r}"

    def test_no_output_file_does_not_save(self, processor, tmp_path):
        folder = self._make_folder(tmp_path)

        with patch.object(processor.file_output, "save_translation_output") as mock_save:
            processor.process_image_folder(str(folder), "English", output_file=None)

        mock_save.assert_not_called()

    def test_routes_through_save_translation_output_not_raw_write(self, processor, tmp_path):
        """Confirm save_translation_output is called (not _save_text_file or raw open)."""
        folder = self._make_folder(tmp_path)
        out = str(tmp_path / "combined.pdf")

        with patch.object(processor.file_output, "save_translation_output") as mock_save:
            processor.process_image_folder(str(folder), "English", output_file=out)

        mock_save.assert_called_once()
        # Third positional argument must be the output file path
        assert mock_save.call_args[0][2] == out
