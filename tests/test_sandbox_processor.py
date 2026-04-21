"""Tests for src/runtime/sandbox_processor.py — SandboxProcessor."""

import os
from typing import Optional, List
from unittest.mock import MagicMock, patch

import pytest

from src.errors import CLIError
from src.models import OutputOptions
from src.runtime.sandbox_processor import SandboxProcessor, _collect_image_files, _parse_page_ranges


# ---------------------------------------------------------------------------
# Helpers — build a SandboxProcessor bypassing real service init
# ---------------------------------------------------------------------------

def _make_processor(monkeypatch) -> SandboxProcessor:
    """Create a SandboxProcessor with all real services replaced by MagicMocks."""
    monkeypatch.setattr("src.runtime.sandbox_processor.get_api_key",
                        lambda name: ("fake-key", "Professor Fake"))

    # Prevent catalog I/O in service constructors
    for svc_path in (
        "src.services.base_service.resolve_model",
        "src.services.base_service.maybe_sync_model_pricing",
        "src.services.base_service.get_model_system_role",
        "src.services.base_service.get_model_max_completion_tokens",
    ):
        monkeypatch.setattr(svc_path, MagicMock(return_value="gpt-4o"), raising=False)

    # Prevent token tracker from touching disk
    monkeypatch.setattr(
        "src.tracking.token_tracker.TokenTracker.__init__",
        lambda self, professor: None,
    )

    proc = SandboxProcessor.__new__(SandboxProcessor)
    proc.professor_name = "fake"
    proc.professor_display_name = "Professor Fake"
    proc.token_tracker = MagicMock()
    proc.token_tracker.usage_data = {"total_usage": {"total_tokens": 0, "total_cost": 0.0}}
    proc.translation_service = MagicMock()
    proc.image_processor_service = MagicMock()
    proc.image_translation_service = MagicMock()
    proc.prompt_service = MagicMock()
    proc.transcription_review_service = MagicMock()
    proc.image_processor = MagicMock()
    proc.pdf_processor = MagicMock()
    proc.file_output = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# SandboxProcessor.__init__
# ---------------------------------------------------------------------------

class TestSandboxProcessorInit:

    def test_init_sets_professor_name(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.runtime.sandbox_processor.get_api_key",
                            lambda name: ("fake-key", "Dr. Smith"))
        # Prevent all service constructors from doing real work
        for cls in (
            "src.runtime.sandbox_processor.TranslationService",
            "src.runtime.sandbox_processor.ImageProcessorService",
            "src.runtime.sandbox_processor.ImageTranslationService",
            "src.runtime.sandbox_processor.PromptService",
            "src.runtime.sandbox_processor.TranscriptionReviewService",
        ):
            monkeypatch.setattr(cls, MagicMock(return_value=MagicMock()), raising=False)

        monkeypatch.setattr("src.runtime.sandbox_processor.TokenTracker",
                            MagicMock(return_value=MagicMock()))

        proc = SandboxProcessor("smith")
        assert proc.professor_name == "smith"
        assert proc.professor_display_name == "Dr. Smith"

    def test_init_raises_cli_error_on_bad_config(self, monkeypatch):
        monkeypatch.setattr("src.runtime.sandbox_processor.get_api_key",
                            lambda name: (_ for _ in ()).throw(ValueError("unknown professor")))
        with pytest.raises(CLIError, match="Configuration error"):
            SandboxProcessor("nobody")


# ---------------------------------------------------------------------------
# _parse_page_ranges
# ---------------------------------------------------------------------------

class TestParsePageRanges:

    def test_none_returns_all_pages(self):
        assert _parse_page_ranges(None) == [(0, None)]

    def test_single_page(self):
        assert _parse_page_ranges("5") == [(4, 4)]

    def test_range(self):
        assert _parse_page_ranges("1-10") == [(0, 9)]

    def test_multi_range(self):
        result = _parse_page_ranges("4,15-17,20")
        assert result == [(3, 3), (14, 16), (19, 19)]

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            _parse_page_ranges("10-5")

    def test_zero_page_raises(self):
        with pytest.raises(ValueError):
            _parse_page_ranges("0")

    def test_zero_in_range_raises(self):
        with pytest.raises(ValueError):
            _parse_page_ranges("0-5")


# ---------------------------------------------------------------------------
# _collect_image_files
# ---------------------------------------------------------------------------

class TestCollectImageFiles:

    def test_returns_sorted_image_paths(self, tmp_path):
        (tmp_path / "b.jpg").write_bytes(b"")
        (tmp_path / "a.png").write_bytes(b"")
        (tmp_path / "z.txt").write_text("")  # not an image
        result = _collect_image_files(str(tmp_path))
        names = [os.path.basename(p) for p in result]
        assert names == ["a.png", "b.jpg"]

    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert _collect_image_files(str(tmp_path)) == []

    def test_non_image_files_excluded(self, tmp_path):
        (tmp_path / "doc.pdf").write_bytes(b"")
        (tmp_path / "img.jpg").write_bytes(b"")
        result = _collect_image_files(str(tmp_path))
        names = [os.path.basename(p) for p in result]
        assert "doc.pdf" not in names
        assert "img.jpg" in names


# ---------------------------------------------------------------------------
# _detect_and_validate_file
# ---------------------------------------------------------------------------

class TestDetectAndValidateFile:

    def test_pdf_detected(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")
        proc.image_processor.is_image_file.return_value = False
        result = proc._detect_and_validate_file(str(f))
        assert result == "pdf"

    def test_docx_detected(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        f = tmp_path / "doc.docx"
        f.write_bytes(b"fake-docx")
        proc.image_processor.is_image_file.return_value = False
        result = proc._detect_and_validate_file(str(f))
        assert result == "docx"

    def test_txt_detected(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        f = tmp_path / "file.txt"
        f.write_text("hello")
        proc.image_processor.is_image_file.return_value = False
        result = proc._detect_and_validate_file(str(f))
        assert result == "txt"

    def test_image_detected(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        f = tmp_path / "scan.jpg"
        f.write_bytes(b"fake-jpg")
        proc.image_processor.is_image_file.return_value = True
        proc.image_processor.validate_image_file.return_value = True
        result = proc._detect_and_validate_file(str(f))
        assert result == "image"

    def test_invalid_image_raises_cli_error(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        f = tmp_path / "bad.jpg"
        f.write_bytes(b"garbage")
        proc.image_processor.is_image_file.return_value = True
        proc.image_processor.validate_image_file.return_value = False
        with pytest.raises(CLIError, match="not valid"):
            proc._detect_and_validate_file(str(f))

    def test_nonexistent_file_raises_cli_error(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        with pytest.raises(CLIError, match="not found"):
            proc._detect_and_validate_file("/no/such/file.txt")

    def test_unsupported_extension_raises_cli_error(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        f = tmp_path / "archive.zip"
        f.write_bytes(b"PK")
        proc.image_processor.is_image_file.return_value = False
        with pytest.raises(CLIError, match="Unsupported"):
            proc._detect_and_validate_file(str(f))


# ---------------------------------------------------------------------------
# _process_text_based_file
# ---------------------------------------------------------------------------

class TestProcessTextBasedFile:

    def test_txt_file_translated(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        f = tmp_path / "source.txt"
        f.write_text("Hello world\n\nSecond paragraph", encoding="utf-8")
        proc.translation_service.translate_text_pages.return_value = ["Bonjour monde"]
        result = proc._process_text_based_file(
            str(f), "txt", None, None, "English", "French",
            OutputOptions(),
        )
        assert result == ["Bonjour monde"]
        proc.translation_service.translate_text_pages.assert_called_once()

    def test_docx_file_translated(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        # Mock DocxProcessor to avoid needing a real docx file
        monkeypatch.setattr(
            "src.runtime.sandbox_processor.DocxProcessor.process_docx_with_pages",
            lambda f, target_page_size: ["Page one text"],
        )
        f = tmp_path / "doc.docx"
        f.write_bytes(b"fake")
        proc.translation_service.translate_text_pages.return_value = ["翻訳"]
        result = proc._process_text_based_file(
            str(f), "docx", None, None, "English", "Japanese",
            OutputOptions(),
        )
        assert result == ["翻訳"]

    def test_unsupported_file_type_raises_value_error(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        f = tmp_path / "file.csv"
        f.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported text file type"):
            proc._process_text_based_file(
                str(f), "csv", None, None, "English", "French",
                OutputOptions(),
            )

    def test_page_range_beyond_document_raises_cli_error(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        monkeypatch.setattr(
            "src.runtime.sandbox_processor.TxtProcessor.process_txt_with_pages",
            lambda f, target_page_size: ["Only one page"],
        )
        f = tmp_path / "file.txt"
        f.write_text("Only one page")
        with pytest.raises(CLIError, match="does not exist"):
            proc._process_text_based_file(
                str(f), "txt", "5", None, "English", "French",
                OutputOptions(),
            )


# ---------------------------------------------------------------------------
# process_prompt / process_transcription_review / _save_text_file
# ---------------------------------------------------------------------------

class TestProcessPrompt:

    def test_delegates_to_prompt_service_and_prints(self, monkeypatch, capsys):
        proc = _make_processor(monkeypatch)
        proc.prompt_service.send_prompt.return_value = "The response text"
        proc.process_prompt("my question")
        out = capsys.readouterr().out
        assert "The response text" in out

    def test_saves_to_output_file(self, monkeypatch, tmp_path, capsys):
        proc = _make_processor(monkeypatch)
        proc.prompt_service.send_prompt.return_value = "saved response"
        out_file = str(tmp_path / "response.txt")
        proc.process_prompt("my question", output_file=out_file)
        assert (tmp_path / "response.txt").read_text() == "saved response"

    def test_exception_wrapped_as_cli_error(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        proc.prompt_service.send_prompt.side_effect = RuntimeError("network fail")
        with pytest.raises(CLIError, match="Error sending prompt"):
            proc.process_prompt("question")


class TestProcessTranscriptionReview:

    def test_delegates_to_service_and_prints(self, monkeypatch, capsys):
        proc = _make_processor(monkeypatch)
        proc.transcription_review_service.review_transcription.return_value = '{"errors": []}'
        proc.process_transcription_review("some text", "Japanese")
        out = capsys.readouterr().out
        assert '{"errors": []}' in out

    def test_saves_to_output_file(self, monkeypatch, tmp_path):
        proc = _make_processor(monkeypatch)
        proc.transcription_review_service.review_transcription.return_value = '{"errors": []}'
        out_file = str(tmp_path / "report.json")
        proc.process_transcription_review("text", "Japanese", output_file=out_file)
        assert (tmp_path / "report.json").read_text() == '{"errors": []}'

    def test_exception_wrapped_as_cli_error(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        proc.transcription_review_service.review_transcription.side_effect = RuntimeError("API fail")
        with pytest.raises(CLIError, match="Error during transcription review"):
            proc.process_transcription_review("text", "Japanese")


class TestSaveTextFile:

    def test_writes_content_to_file(self, tmp_path):
        out = tmp_path / "output.txt"
        SandboxProcessor._save_text_file("hello world", str(out))
        assert out.read_text() == "hello world"

    def test_prints_saved_path(self, tmp_path, capsys):
        out = tmp_path / "output.txt"
        SandboxProcessor._save_text_file("content", str(out), label="Translation")
        captured = capsys.readouterr().out
        assert "Translation" in captured
        assert "output.txt" in captured


# ---------------------------------------------------------------------------
# process_image / process_image_folder
# ---------------------------------------------------------------------------

class TestProcessImage:

    def test_delegates_to_image_processor_service(self, monkeypatch, capsys):
        proc = _make_processor(monkeypatch)
        proc.image_processor_service.process_image_ocr.return_value = "extracted text"
        proc.process_image("/fake/path/img.jpg", "English")
        out = capsys.readouterr().out
        assert "extracted text" in out

    def test_saves_to_output_file(self, monkeypatch, tmp_path):
        proc = _make_processor(monkeypatch)
        proc.image_processor_service.process_image_ocr.return_value = "ocr result"
        out_file = str(tmp_path / "result.txt")
        proc.process_image("/fake/img.jpg", "English", output_file=out_file)
        proc.file_output.save_translation_output.assert_called_once()

    def test_exception_wrapped_as_cli_error(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        proc.image_processor_service.process_image_ocr.side_effect = RuntimeError("vision fail")
        with pytest.raises(CLIError, match="Error processing image"):
            proc.process_image("/fake/img.jpg", "English")


class TestProcessImageFolder:

    def test_empty_folder_raises_cli_error(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        with pytest.raises(CLIError, match="No image files found"):
            proc.process_image_folder(str(tmp_path), "English")

    def test_sequential_processing_prints_results(self, tmp_path, monkeypatch, capsys):
        proc = _make_processor(monkeypatch)
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"fake-jpg")
        proc.image_processor_service.process_image_ocr.return_value = "OCR output"
        proc.process_image_folder(str(tmp_path), "English")
        out = capsys.readouterr().out
        assert "scan.jpg" in out
        assert "OCR output" in out

    def test_saves_output_file_after_processing(self, tmp_path, monkeypatch):
        proc = _make_processor(monkeypatch)
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"fake")
        proc.image_processor_service.process_image_ocr.return_value = "text"
        proc.process_image_folder(str(tmp_path), "English", output_file="out.txt")
        proc.file_output.save_translation_output.assert_called_once()


# ---------------------------------------------------------------------------
# process_image_translation
# ---------------------------------------------------------------------------

class TestProcessImageTranslation:

    def test_prints_transcript_and_translation(self, monkeypatch, capsys):
        proc = _make_processor(monkeypatch)
        proc.image_translation_service.process_image_translation.return_value = (
            "OCR transcript", "Translated text"
        )
        proc.process_image_translation("/fake/img.jpg", "Chinese", "English")
        out = capsys.readouterr().out
        assert "OCR transcript" in out
        assert "Translated text" in out

    def test_saves_translation_when_output_file_set(self, monkeypatch, tmp_path):
        proc = _make_processor(monkeypatch)
        proc.image_translation_service.process_image_translation.return_value = (
            "", "Translation only"
        )
        opts = OutputOptions(output_file=str(tmp_path / "out.txt"))
        proc.process_image_translation("/fake/img.jpg", "Chinese", "English", opts)
        proc.file_output.save_translation_output.assert_called_once()

    def test_no_transcript_skips_transcript_section(self, monkeypatch, capsys):
        proc = _make_processor(monkeypatch)
        proc.image_translation_service.process_image_translation.return_value = (
            None, "Translation"
        )
        proc.process_image_translation("/fake/img.jpg", "Japanese", "English")
        out = capsys.readouterr().out
        assert "Transcript" not in out


# ---------------------------------------------------------------------------
# translate_custom_text
# ---------------------------------------------------------------------------

class TestTranslateCustomText:

    def test_translates_and_calls_service(self, monkeypatch, capsys):
        proc = _make_processor(monkeypatch)
        inputs = iter(["Hello world", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        proc.translation_service.translate_text.return_value = "Bonjour monde"
        proc.translate_custom_text("English", "French")
        proc.translation_service.translate_text.assert_called_once_with(
            "Hello world", "English", "French"
        )

    def test_empty_text_returns_without_error(self, monkeypatch, capsys):
        proc = _make_processor(monkeypatch)
        inputs = iter(["---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        proc.translate_custom_text("English", "French")
        out = capsys.readouterr().out
        assert "No text provided" in out

    def test_keyboard_interrupt_prints_cancelled(self, monkeypatch, capsys):
        proc = _make_processor(monkeypatch)

        def raise_interrupt(*_):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", raise_interrupt)
        proc.translate_custom_text("English", "French")
        out = capsys.readouterr().out
        assert "cancelled" in out.lower()

    def test_with_abstract_uses_translate_page_text(self, monkeypatch, capsys):
        proc = _make_processor(monkeypatch)
        # First _collect_multiline is for abstract, second for text
        call_count = [0]
        def fake_input():
            call_count[0] += 1
            if call_count[0] <= 2:
                return "abstract text" if call_count[0] == 1 else "---"
            if call_count[0] == 3:
                return "translate this"
            return "---"
        monkeypatch.setattr("builtins.input", fake_input)
        proc.translation_service.translate_page_text.return_value = "结果"
        proc.translate_custom_text("English", "Chinese", abstract=True)
        proc.translation_service.translate_page_text.assert_called_once()


# ---------------------------------------------------------------------------
# translate_document — ImportError for python-docx (lines 301-302)
# ---------------------------------------------------------------------------

class TestTranslateDocumentImportError:

    def test_missing_python_docx_raises_cli_error(self, monkeypatch, tmp_path):
        """When DocxProcessor raises ImportError mentioning python-docx, CLIError wraps it."""
        proc = _make_processor(monkeypatch)
        docx_file = tmp_path / "doc.docx"
        docx_file.write_bytes(b"fake docx content")

        monkeypatch.setattr(
            "src.runtime.sandbox_processor.SandboxProcessor._detect_and_validate_file",
            lambda self, fp: "docx"
        )

        # Patch _process_text_based_file to raise ImportError as python-docx would
        def raise_import(*a, **kw):
            raise ImportError("No module named 'python-docx'")

        monkeypatch.setattr(
            "src.runtime.sandbox_processor.SandboxProcessor._process_text_based_file",
            raise_import,
        )

        with pytest.raises(CLIError, match="pip install python-docx"):
            proc.translate_document(str(docx_file), "Chinese", "English")


# ---------------------------------------------------------------------------
# translate_document — image file path (lines 193-207)
# ---------------------------------------------------------------------------

class TestTranslateDocumentImagePath:

    def test_image_file_delegates_to_process_image_translation(self, monkeypatch, tmp_path):
        proc = _make_processor(monkeypatch)
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake image")

        monkeypatch.setattr(
            "src.runtime.sandbox_processor.SandboxProcessor._detect_and_validate_file",
            lambda self, fp: "image"
        )
        proc.process_image_translation = MagicMock()
        proc.translate_document(str(img), "Chinese", "English")
        proc.process_image_translation.assert_called_once()

    def test_image_file_exception_raises_cli_error(self, monkeypatch, tmp_path):
        proc = _make_processor(monkeypatch)
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake image")

        monkeypatch.setattr(
            "src.runtime.sandbox_processor.SandboxProcessor._detect_and_validate_file",
            lambda self, fp: "image"
        )
        proc.process_image_translation = MagicMock(side_effect=RuntimeError("API down"))
        with pytest.raises(CLIError, match="API down"):
            proc.translate_document(str(img), "Chinese", "English")


# ---------------------------------------------------------------------------
# translate_document — save output path (lines 249-260)
# ---------------------------------------------------------------------------

class TestTranslateDocumentSaveOutput:

    def test_saves_output_when_output_file_specified(self, monkeypatch, tmp_path):
        proc = _make_processor(monkeypatch)
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("Hello content", encoding="utf-8")

        monkeypatch.setattr(
            "src.runtime.sandbox_processor.SandboxProcessor._detect_and_validate_file",
            lambda self, fp: "txt"
        )
        monkeypatch.setattr(
            "src.runtime.sandbox_processor.SandboxProcessor._process_text_based_file",
            lambda *a, **kw: ["Translated text"]
        )

        out_file = str(tmp_path / "out.txt")
        opts = OutputOptions(output_file=out_file)
        proc.translate_document(str(txt_file), "Chinese", "English", opts=opts)
        proc.file_output.save_translation_output.assert_called_once()


# ---------------------------------------------------------------------------
# translate_custom_text — save output path (lines 300-301)
# ---------------------------------------------------------------------------

class TestTranslateCustomTextSaveOutput:

    def test_saves_output_when_output_file_specified(self, monkeypatch, tmp_path, capsys):
        proc = _make_processor(monkeypatch)
        inputs = iter(["Translate this", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))

        proc.translation_service.translate_text.return_value = "翻訳結果"
        out_file = str(tmp_path / "result.txt")
        opts = OutputOptions(output_file=out_file)
        proc.translate_custom_text("English", "Japanese", opts=opts)
        proc.file_output.save_translation_output.assert_called_once()


# ---------------------------------------------------------------------------
# process_image_translation_folder — parallel path (lines 371-457)
# ---------------------------------------------------------------------------

class TestProcessImageTranslationFolderParallel:

    def test_parallel_path_processes_images(self, monkeypatch, tmp_path, capsys):
        proc = _make_processor(monkeypatch)
        folder = tmp_path / "imgs"
        folder.mkdir()
        # Create 2 fake images
        for name in ["a.jpg", "b.jpg"]:
            (folder / name).write_bytes(b"fake")

        proc.image_translation_service.process_image_translation = MagicMock(
            return_value=("transcript", "translation")
        )
        proc.image_translation_service._get_model = MagicMock(return_value="gpt-4o")
        proc.image_translation_service._suppress_inline_print = False

        proc.process_image_translation_folder(
            str(folder), "Chinese", "English", OutputOptions(), workers=2
        )
        # Both images should have been processed
        assert proc.image_translation_service.process_image_translation.call_count == 2

    def test_parallel_worker_exception_stored_as_error(self, monkeypatch, tmp_path, capsys):
        proc = _make_processor(monkeypatch)
        folder = tmp_path / "err_imgs"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(b"fake")

        proc.image_translation_service.process_image_translation = MagicMock(
            side_effect=RuntimeError("worker failed")
        )
        proc.image_translation_service._get_model = MagicMock(return_value="gpt-4o")
        proc.image_translation_service._suppress_inline_print = False

        # Should not raise — errors collected and printed
        proc.process_image_translation_folder(
            str(folder), "Chinese", "English", OutputOptions(), workers=2
        )
        out = capsys.readouterr().out
        assert "worker failed" in out or "Error" in out


# ---------------------------------------------------------------------------
# process_image_folder — parallel path (lines 481-end)
# ---------------------------------------------------------------------------

class TestProcessImageFolderParallel:

    def test_parallel_ocr_processes_all_images(self, monkeypatch, tmp_path, capsys):
        proc = _make_processor(monkeypatch)
        folder = tmp_path / "ocr_imgs"
        folder.mkdir()
        for name in ["x.jpg", "y.jpg"]:
            (folder / name).write_bytes(b"fake")

        proc.image_processor_service.process_image_ocr = MagicMock(
            return_value="Extracted text"
        )

        proc.process_image_folder(str(folder), "English", workers=2)
        assert proc.image_processor_service.process_image_ocr.call_count == 2
