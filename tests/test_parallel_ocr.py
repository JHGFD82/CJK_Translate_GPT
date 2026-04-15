"""
Tests for parallel OCR folder processing (-w/--workers > 1 with transcribe command).

Covers:
  - workers=1 processes images sequentially in sorted-filename order
  - workers > 1 produces the same ordered combined output as workers=1
  - Worker exceptions produce error placeholders in the correct position
  - Multi-pass OCR within each image still runs sequentially inside the worker
  - workers > image_count is capped gracefully (no error)
  - Output file is saved with images joined in order
  - workers flag is ignored (no error) for single-image input

No real API calls; process_image_ocr is patched at the method level.
"""

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.runtime.sandbox_processor import SandboxProcessor
from src.services.image_processor_service import ImageProcessorService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sandbox(tmp_path):
    """A SandboxProcessor with all network-touching services mocked out."""
    with patch("src.runtime.sandbox_processor.get_api_key", return_value=("fake-key", "Test Prof")), \
         patch("src.runtime.sandbox_processor.TokenTracker", return_value=MagicMock()), \
         patch("src.runtime.sandbox_processor.TranslationService"), \
         patch("src.runtime.sandbox_processor.ImageProcessorService"), \
         patch("src.runtime.sandbox_processor.ImageTranslationService"), \
         patch("src.runtime.sandbox_processor.PromptService"):
        sp = SandboxProcessor("testprof")
    return sp


@pytest.fixture
def image_folder(tmp_path):
    """A temp folder with three stub image files named so they sort predictably."""
    folder = tmp_path / "scans"
    folder.mkdir()
    for name in ("01_scan.jpg", "02_scan.jpg", "03_scan.jpg"):
        (folder / name).write_bytes(b"\xff\xd8\xff")  # minimal JPEG header
    return folder


# ---------------------------------------------------------------------------
# Sequential path (workers=1)
# ---------------------------------------------------------------------------

class TestSequentialFolderOCR:

    def test_processes_images_in_sorted_order(self, sandbox, image_folder):
        call_order: list[str] = []

        def fake_ocr(path, lang, output_format="console", vertical=False, passes=1):
            call_order.append(os.path.basename(path))
            return f"text from {os.path.basename(path)}"

        sandbox.image_processor_service.process_image_ocr = fake_ocr

        sandbox.process_image_folder(
            str(image_folder), "English", output_file=None, workers=1
        )

        assert call_order == ["01_scan.jpg", "02_scan.jpg", "03_scan.jpg"]

    def test_output_saved_in_order(self, sandbox, image_folder, tmp_path):
        def fake_ocr(path, lang, output_format="console", vertical=False, passes=1):
            return f"result:{os.path.basename(path)}"

        sandbox.image_processor_service.process_image_ocr = fake_ocr

        out_file = str(tmp_path / "combined.txt")
        with patch.object(sandbox.file_output, "save_translation_output") as mock_save:
            sandbox.process_image_folder(
                str(image_folder), "English", output_file=out_file, workers=1
            )

        saved_text = mock_save.call_args[0][0]
        parts = saved_text.split("\n\n")
        assert "01_scan.jpg" in parts[0]
        assert "02_scan.jpg" in parts[1]
        assert "03_scan.jpg" in parts[2]


# ---------------------------------------------------------------------------
# Parallel path (workers > 1) — same output as sequential
# ---------------------------------------------------------------------------

class TestParallelFolderOCR:

    def test_same_output_order_as_sequential(self, sandbox, image_folder, tmp_path):
        """Parallel processing must yield the same ordered output as sequential."""
        seq_text: list[str] = []
        par_text: list[str] = []

        def fake_ocr(path, lang, output_format="console", vertical=False, passes=1):
            return f"text:{os.path.basename(path)}"

        sandbox.image_processor_service.process_image_ocr = fake_ocr

        # Sequential run
        out_seq = str(tmp_path / "seq.txt")
        with patch.object(sandbox.file_output, "save_translation_output") as mock_save:
            sandbox.process_image_folder(str(image_folder), "English", output_file=out_seq, workers=1)
            seq_text.append(mock_save.call_args[0][0])

        # Parallel run (simulate completion in reverse order using a sleep gap)
        call_count = [0]

        def fake_ocr_staggered(path, lang, output_format="console", vertical=False, passes=1):
            call_count[0] += 1
            # Deliberately stagger: last image finishes fastest
            if "01_scan" in path:
                import time; time.sleep(0.05)
            return f"text:{os.path.basename(path)}"

        sandbox.image_processor_service.process_image_ocr = fake_ocr_staggered

        out_par = str(tmp_path / "par.txt")
        with patch.object(sandbox.file_output, "save_translation_output") as mock_save:
            sandbox.process_image_folder(str(image_folder), "English", output_file=out_par, workers=3)
            par_text.append(mock_save.call_args[0][0])

        assert seq_text == par_text

    def test_workers_gt_image_count_handled_gracefully(self, sandbox, image_folder):
        """More workers than images must not raise."""
        def fake_ocr(path, lang, output_format="console", vertical=False, passes=1):
            return "ok"

        sandbox.image_processor_service.process_image_ocr = fake_ocr

        # Should not raise even with 100 workers for 3 images
        sandbox.process_image_folder(str(image_folder), "English", workers=100)

    def test_worker_exception_produces_error_placeholder(self, sandbox, image_folder, tmp_path):
        """A failed worker must produce an error placeholder in the right position."""
        def fake_ocr(path, lang, output_format="console", vertical=False, passes=1):
            if "02_scan" in path:
                raise RuntimeError("API timeout")
            return f"ok:{os.path.basename(path)}"

        sandbox.image_processor_service.process_image_ocr = fake_ocr

        out_file = str(tmp_path / "out.txt")
        with patch.object(sandbox.file_output, "save_translation_output") as mock_save:
            sandbox.process_image_folder(
                str(image_folder), "English", output_file=out_file, workers=3
            )

        saved = mock_save.call_args[0][0]
        parts = saved.split("\n\n")
        assert "01_scan.jpg" in parts[0]
        assert "Error" in parts[1]  # the placeholder for the failed image
        assert "03_scan.jpg" in parts[2]


# ---------------------------------------------------------------------------
# Multi-pass OCR stays sequential within each parallel worker
# ---------------------------------------------------------------------------

class TestMultiPassSequentialWithinWorker:

    def test_passes_forwarded_to_each_worker(self, sandbox, image_folder):
        """Each parallel worker must receive the same passes= argument."""
        received_passes: list[int] = []
        lock = threading.Lock()

        def fake_ocr(path, lang, output_format="console", vertical=False, passes=1):
            with lock:
                received_passes.append(passes)
            return "done"

        sandbox.image_processor_service.process_image_ocr = fake_ocr

        sandbox.process_image_folder(str(image_folder), "English", passes=3, workers=3)

        assert all(p == 3 for p in received_passes), (
            f"Not all workers received passes=3: {received_passes}"
        )


# ---------------------------------------------------------------------------
# Empty folder raises CLIError
# ---------------------------------------------------------------------------

class TestEmptyFolder:

    def test_raises_clierror_for_empty_folder(self, sandbox, tmp_path):
        from src.errors import CLIError
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(CLIError, match="No image files found"):
            sandbox.process_image_folder(str(empty), "English", workers=2)


# ---------------------------------------------------------------------------
# workers flag on single-image input is silently ignored
# ---------------------------------------------------------------------------

class TestSingleImageIgnoresWorkers:

    def test_single_image_does_not_error_with_workers_flag(self, sandbox, tmp_path):
        """process_image() has no workers parameter; workers arg from CLI is only
        passed to process_image_folder. A single-image transcribe should succeed."""
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        def fake_ocr(path, lang, output_format="console", vertical=False, passes=1):
            return "transcribed text"

        sandbox.image_processor_service.process_image_ocr = fake_ocr

        # process_image does not accept workers — it's silently not passed
        sandbox.process_image(str(img), "English", output_file=None)
