"""
Tests for document and image processor utilities (no API calls):
  - TxtProcessor.extract_raw_content, process_txt_with_pages
  - DocxProcessor.extract_raw_content, process_docx_with_pages
  - ImageProcessor.is_image_file, validate_image_file, local_image_to_data_url
"""

import base64
import io
import struct
import zlib

import pytest

from src.processors.txt_processor import TxtProcessor
from src.processors.docx_processor import DocxProcessor
from src.processors.image_processor import ImageProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_docx_bytes(paragraphs: list[str]) -> io.BytesIO:
    """Create a minimal real .docx in memory using python-docx."""
    from docx import Document
    doc = Document()
    for para in paragraphs:
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def _make_tiny_png() -> bytes:
    """Return the bytes of a 1×1 red PNG (no external dependencies)."""
    # PNG signature
    sig = b'\x89PNG\r\n\x1a\n'

    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = struct.pack('>I', len(data)) + ctype + data
        return c + struct.pack('>I', zlib.crc32(ctype + data) & 0xFFFFFFFF)

    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
    # 1×1 RGB pixel (255,0,0) = red; filter byte 0 prepended
    raw = b'\x00\xff\x00\x00'
    idat = chunk(b'IDAT', zlib.compress(raw))
    iend = chunk(b'IEND', b'')
    return sig + ihdr + idat + iend


# ---------------------------------------------------------------------------
# TxtProcessor
# ---------------------------------------------------------------------------


class TestTxtProcessor:

    def _file(self, content: str) -> io.StringIO:
        return io.StringIO(content)

    def test_extract_raw_content_returns_stripped_text(self):
        p = TxtProcessor()
        result = p.extract_raw_content(self._file("  Hello world  "))
        assert result == "Hello world"

    def test_extract_raw_content_empty_file(self):
        p = TxtProcessor()
        result = p.extract_raw_content(self._file(""))
        assert result == ""

    def test_extract_raw_content_preserves_cjk(self):
        p = TxtProcessor()
        result = p.extract_raw_content(self._file("日本語テキスト"))
        assert result == "日本語テキスト"

    def test_process_txt_with_pages_single_page(self):
        pages = TxtProcessor.process_txt_with_pages(self._file("Short content"))
        assert len(pages) >= 1
        assert "Short content" in pages[0]

    def test_process_txt_with_pages_empty_file_returns_empty_string(self):
        pages = TxtProcessor.process_txt_with_pages(self._file(""))
        assert pages == [""]

    def test_process_txt_with_pages_splits_large_content(self):
        # Create content that will definitely exceed a small page size
        big_content = "\n\n".join(["Paragraph " + str(i) * 100 for i in range(20)])
        pages = TxtProcessor.process_txt_with_pages(
            io.StringIO(big_content), target_page_size=200
        )
        assert len(pages) > 1

    def test_process_txt_with_pages_preserves_all_content(self):
        content = "First paragraph\n\nSecond paragraph\n\nThird paragraph"
        pages = TxtProcessor.process_txt_with_pages(io.StringIO(content))
        combined = "\n\n".join(pages)
        assert "First paragraph" in combined
        assert "Second paragraph" in combined
        assert "Third paragraph" in combined

    def test_process_txt_with_pages_raises_on_read_error(self):
        bad_file = io.StringIO()
        bad_file.close()  # Closed → reading will raise ValueError
        with pytest.raises(Exception, match="Failed to process"):
            TxtProcessor.process_txt_with_pages(bad_file)


# ---------------------------------------------------------------------------
# DocxProcessor
# ---------------------------------------------------------------------------


class TestDocxProcessor:

    def test_extract_raw_content_returns_paragraph_text(self):
        buf = _make_docx_bytes(["Hello", "World"])
        p = DocxProcessor()
        result = p.extract_raw_content(buf)
        assert "Hello" in result
        assert "World" in result

    def test_extract_raw_content_empty_document_returns_empty_string(self):
        buf = _make_docx_bytes([])
        p = DocxProcessor()
        result = p.extract_raw_content(buf)
        assert result == ""

    def test_extract_raw_content_preserves_cjk(self):
        buf = _make_docx_bytes(["日本語テキスト", "中文内容"])
        p = DocxProcessor()
        result = p.extract_raw_content(buf)
        assert "日本語テキスト" in result
        assert "中文内容" in result

    def test_extract_raw_content_paragraphs_joined_with_double_newline(self):
        buf = _make_docx_bytes(["Para1", "Para2"])
        p = DocxProcessor()
        result = p.extract_raw_content(buf)
        assert "\n\n" in result

    def test_process_docx_with_pages_single_page(self):
        buf = _make_docx_bytes(["Short text"])
        pages = DocxProcessor.process_docx_with_pages(buf)
        assert len(pages) >= 1
        assert "Short text" in pages[0]

    def test_process_docx_with_pages_empty_doc_returns_empty_string(self):
        buf = _make_docx_bytes([])
        pages = DocxProcessor.process_docx_with_pages(buf)
        assert pages == [""]

    def test_process_docx_with_pages_splits_large_content(self):
        big_paragraphs = ["Paragraph " + str(i) * 80 for i in range(15)]
        buf = _make_docx_bytes(big_paragraphs)
        pages = DocxProcessor.process_docx_with_pages(buf, target_page_size=200)
        assert len(pages) > 1

    def test_process_docx_with_pages_preserves_all_paragraphs(self):
        buf = _make_docx_bytes(["Alpha", "Beta", "Gamma"])
        pages = DocxProcessor.process_docx_with_pages(buf)
        combined = "\n\n".join(pages)
        for word in ("Alpha", "Beta", "Gamma"):
            assert word in combined


# ---------------------------------------------------------------------------
# ImageProcessor
# ---------------------------------------------------------------------------


class TestImageProcessor:

    # --- is_image_file -------------------------------------------------------

    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"])
    def test_recognized_extensions_return_true(self, ext):
        assert ImageProcessor.is_image_file(f"photo{ext}") is True

    @pytest.mark.parametrize("ext", [".pdf", ".docx", ".txt", ".mp4", ""])
    def test_non_image_extensions_return_false(self, ext):
        assert ImageProcessor.is_image_file(f"file{ext}") is False

    def test_uppercase_extension_recognized(self):
        assert ImageProcessor.is_image_file("PHOTO.JPG") is True

    def test_mixed_case_extension_recognized(self):
        assert ImageProcessor.is_image_file("scan.Png") is True

    # --- validate_image_file -------------------------------------------------

    def test_valid_image_file_returns_true(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(_make_tiny_png())
        assert ImageProcessor.validate_image_file(str(img)) is True

    def test_missing_file_returns_false(self, tmp_path):
        assert ImageProcessor.validate_image_file(str(tmp_path / "missing.jpg")) is False

    def test_wrong_extension_returns_false(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        assert ImageProcessor.validate_image_file(str(f)) is False

    # --- local_image_to_data_url ---------------------------------------------

    def test_returns_data_url_format(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(_make_tiny_png())
        result = ImageProcessor().local_image_to_data_url(str(img))
        assert result.startswith("data:image/png;base64,")

    def test_base64_payload_is_valid(self, tmp_path):
        img = tmp_path / "photo.png"
        png_bytes = _make_tiny_png()
        img.write_bytes(png_bytes)
        result = ImageProcessor().local_image_to_data_url(str(img))
        payload = result.split(",", 1)[1]
        decoded = base64.b64decode(payload)
        assert decoded == png_bytes

    def test_jpeg_uses_jpeg_mime_type(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)  # minimal JPEG header
        result = ImageProcessor().local_image_to_data_url(str(img))
        assert "image/jpeg" in result

    def test_unknown_mime_type_falls_back_to_octet_stream(self, tmp_path):
        # Use an extension that has no registered MIME type on any platform
        img = tmp_path / "photo.zzunknownzz"
        img.write_bytes(b"\x00\x01\x02\x03")
        result = ImageProcessor().local_image_to_data_url(str(img))
        assert "application/octet-stream" in result
