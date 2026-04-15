"""
Tests for parallel page translation (-w/--workers > 1).

Covers:
  - Pages assembled in correct index order regardless of future completion order
  - Each worker receives previous_translated="" and the correct untranslated previous_page
  - Temp files are cleaned up on success and on worker exception
  - workers=1 uses the sequential path (ThreadPoolExecutor is NOT instantiated)
  - workers > page_count is capped gracefully
  - progressive_save=True + workers > 1 emits a warning and still returns results
  - Context-length splitting (generate_text deque loop) still works inside each worker

No real API calls are made; generate_text and translate_page_text are patched.
"""

import threading
from unittest.mock import MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.services.translation_service import TranslationService
from src.models.output_options import OutputOptions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def svc():
    """TranslationService with mocked token tracker."""
    return TranslationService(api_key="fake-key", token_tracker=MagicMock())


def _make_triples(n: int):
    """Return n (index, page_text, previous_page) triples with distinct content."""
    triples = []
    prev = ""
    for i in range(n):
        page = f"Page {i} content"
        triples.append((i, page, prev))
        prev = page
    return triples


# ---------------------------------------------------------------------------
# _translate_pages_parallel — order correctness
# ---------------------------------------------------------------------------

class TestParallelOrder:

    def test_results_are_in_page_index_order(self, svc):
        """Even if futures complete in reverse order the output list preserves page sequence."""
        completion_order = []

        def fake_generate(abstract, page_text, prev_page, page_num, src, tgt, fmt, previous_translated):
            # Simulate the last page finishing first
            if page_num == 2:
                import time; time.sleep(0.01)  # last page is actually slowest
            result = f"\n\n-- Page {page_num + 1} -- \n\nTranslated: {page_text}"
            completion_order.append(page_num)
            return result

        all_triples = _make_triples(3)
        with patch.object(svc, "generate_text", side_effect=fake_generate):
            results = svc._translate_pages_parallel(
                all_triples,
                abstract_text="",
                source_language="Chinese",
                target_language="English",
                output_format="console",
                unit_label="page",
                workers=3,
                opts=OutputOptions(),
            )

        assert len(results) == 3
        # Results must be in page order regardless of completion order
        assert "Page 0 content" in results[0]
        assert "Page 1 content" in results[1]
        assert "Page 2 content" in results[2]

    def test_single_page_document(self, svc):
        all_triples = _make_triples(1)

        def fake_generate(abstract, page_text, prev_page, page_num, src, tgt, fmt, previous_translated):
            return f"\n\n-- Page 1 -- \n\n{page_text}"

        with patch.object(svc, "generate_text", side_effect=fake_generate):
            results = svc._translate_pages_parallel(
                all_triples,
                abstract_text="",
                source_language="Japanese",
                target_language="English",
                output_format="console",
                unit_label="page",
                workers=4,  # more workers than pages
                opts=OutputOptions(),
            )
        assert len(results) == 1
        assert "Page 0 content" in results[0]


# ---------------------------------------------------------------------------
# _translate_pages_parallel — context passed to workers
# ---------------------------------------------------------------------------

class TestParallelContext:

    def test_previous_translated_is_empty_string_for_every_worker(self, svc):
        """In parallel mode, no worker should receive a prior translation as context."""
        received_previous_translated = []

        def fake_generate(abstract, page_text, prev_page, page_num, src, tgt, fmt, previous_translated):
            received_previous_translated.append(previous_translated)
            return f"translated {page_num}"

        all_triples = _make_triples(4)
        with patch.object(svc, "generate_text", side_effect=fake_generate):
            svc._translate_pages_parallel(
                all_triples,
                abstract_text="",
                source_language="Chinese",
                target_language="English",
                output_format="console",
                unit_label="page",
                workers=4,
                opts=OutputOptions(),
            )

        assert all(pt == "" for pt in received_previous_translated), (
            f"Some workers received non-empty previous_translated: {received_previous_translated}"
        )

    def test_previous_page_source_text_is_passed_correctly(self, svc):
        """Each worker receives the untranslated source text of the prior page."""
        received_prev_pages = {}

        def fake_generate(abstract, page_text, prev_page, page_num, src, tgt, fmt, previous_translated):
            received_prev_pages[page_num] = prev_page
            return f"translated {page_num}"

        all_triples = _make_triples(3)
        with patch.object(svc, "generate_text", side_effect=fake_generate):
            svc._translate_pages_parallel(
                all_triples,
                abstract_text="",
                source_language="Chinese",
                target_language="English",
                output_format="console",
                unit_label="page",
                workers=3,
                opts=OutputOptions(),
            )

        assert received_prev_pages[0] == ""          # first page has no previous
        assert received_prev_pages[1] == "Page 0 content"
        assert received_prev_pages[2] == "Page 1 content"

    def test_abstract_text_passed_to_every_worker(self, svc):
        received_abstracts = []

        def fake_generate(abstract, page_text, prev_page, page_num, src, tgt, fmt, previous_translated):
            received_abstracts.append(abstract)
            return f"translated {page_num}"

        all_triples = _make_triples(3)
        with patch.object(svc, "generate_text", side_effect=fake_generate):
            svc._translate_pages_parallel(
                all_triples,
                abstract_text="The abstract of this paper",
                source_language="Chinese",
                target_language="English",
                output_format="console",
                unit_label="page",
                workers=3,
                opts=OutputOptions(),
            )

        assert all(a == "The abstract of this paper" for a in received_abstracts)


# ---------------------------------------------------------------------------
# _translate_pages_parallel — worker count capping
# ---------------------------------------------------------------------------

class TestWorkerCapping:

    def test_workers_capped_at_page_count(self, svc):
        """When workers > pages, the executor is capped and no error is raised."""
        all_triples = _make_triples(2)

        created_pools = []
        real_TPool = ThreadPoolExecutor

        def spy_pool(max_workers=None, **kwargs):
            created_pools.append(max_workers)
            return real_TPool(max_workers=max_workers, **kwargs)

        def fake_generate(abstract, page_text, prev_page, page_num, src, tgt, fmt, previous_translated):
            return f"ok {page_num}"

        with patch.object(svc, "generate_text", side_effect=fake_generate):
            with patch(
                "src.services.translation_service.ThreadPoolExecutor",
                side_effect=spy_pool,
            ):
                results = svc._translate_pages_parallel(
                    all_triples,
                    abstract_text="",
                    source_language="Chinese",
                    target_language="English",
                    output_format="console",
                    unit_label="page",
                    workers=10,   # far more than 2 pages
                    opts=OutputOptions(),
                )

        assert len(results) == 2
        # The pool must be created with at most len(all_triples) workers
        assert created_pools[0] <= 2


# ---------------------------------------------------------------------------
# _translate_pages_parallel — temp file cleanup
# ---------------------------------------------------------------------------

class TestTempFileCleanup:

    def test_tmpdir_deleted_on_success(self, svc):
        """All temp files and the temp directory are removed after a successful run."""
        import os
        created_tmpdirs = []
        real_mkdtemp = __import__("tempfile").mkdtemp

        def spy_mkdtemp(**kwargs):
            d = real_mkdtemp(**kwargs)
            created_tmpdirs.append(d)
            return d

        def fake_generate(abstract, page_text, prev_page, page_num, src, tgt, fmt, previous_translated):
            return f"done {page_num}"

        with patch("src.services.translation_service.tempfile.mkdtemp", side_effect=spy_mkdtemp):
            with patch.object(svc, "generate_text", side_effect=fake_generate):
                svc._translate_pages_parallel(
                    _make_triples(3),
                    abstract_text="",
                    source_language="Chinese",
                    target_language="English",
                    output_format="console",
                    unit_label="page",
                    workers=2,
                    opts=OutputOptions(),
                )

        assert len(created_tmpdirs) == 1
        assert not os.path.exists(created_tmpdirs[0]), "Temp directory should be deleted after success"

    def test_tmpdir_deleted_on_worker_exception(self, svc):
        """Temp directory is cleaned up even when a worker raises an exception."""
        import os
        created_tmpdirs = []
        real_mkdtemp = __import__("tempfile").mkdtemp

        def spy_mkdtemp(**kwargs):
            d = real_mkdtemp(**kwargs)
            created_tmpdirs.append(d)
            return d

        call_count = [0]

        def failing_generate(abstract, page_text, prev_page, page_num, src, tgt, fmt, previous_translated):
            call_count[0] += 1
            if page_num == 1:
                raise RuntimeError("Simulated API failure")
            return f"done {page_num}"

        with patch("src.services.translation_service.tempfile.mkdtemp", side_effect=spy_mkdtemp):
            with patch.object(svc, "generate_text", side_effect=failing_generate):
                results = svc._translate_pages_parallel(
                    _make_triples(3),
                    abstract_text="",
                    source_language="Chinese",
                    target_language="English",
                    output_format="console",
                    unit_label="page",
                    workers=2,
                    opts=OutputOptions(),
                )

        assert len(created_tmpdirs) == 1
        assert not os.path.exists(created_tmpdirs[0]), "Temp directory should be deleted even after worker failure"
        # The failed page gets an error placeholder, others succeed
        assert len(results) == 3
        assert "Translation error" in results[1]


# ---------------------------------------------------------------------------
# _translate_page_sequence — sequential vs parallel dispatch
# ---------------------------------------------------------------------------

class TestSequentialVsParallelDispatch:

    def test_workers_1_does_not_instantiate_threadpoolexecutor(self, svc):
        """workers=1 must use the sequential code path with no ThreadPoolExecutor."""
        def fake_generate(abstract, page_text, prev_page, page_num, src, tgt, fmt, previous_translated):
            return f"seq {page_num}"

        with patch.object(svc, "generate_text", side_effect=fake_generate):
            with patch("src.services.translation_service.ThreadPoolExecutor") as mock_pool:
                svc._translate_page_sequence(
                    iter(_make_triples(3)),
                    abstract_text="",
                    source_language="Chinese",
                    target_language="English",
                    output_format="console",
                    first_index=0,
                    unit_label="page",
                    opts=OutputOptions(),
                    input_file_path=None,
                    workers=1,
                )

        mock_pool.assert_not_called()

    def test_workers_gt_1_calls_parallel_path(self, svc):
        """workers > 1 delegates to _translate_pages_parallel."""
        with patch.object(svc, "_translate_pages_parallel", return_value=["p0", "p1"]) as mock_parallel:
            result = svc._translate_page_sequence(
                iter(_make_triples(2)),
                abstract_text="",
                source_language="Chinese",
                target_language="English",
                output_format="console",
                first_index=0,
                unit_label="page",
                opts=OutputOptions(),
                input_file_path=None,
                workers=2,
            )

        mock_parallel.assert_called_once()
        assert result == ["p0", "p1"]


# ---------------------------------------------------------------------------
# progressive_save + workers > 1 warning
# ---------------------------------------------------------------------------

class TestProgressiveSaveWarning:

    def test_warning_printed_when_progressive_save_and_workers_gt_1(self, svc, capsys):
        """A warning is printed when progressive_save and workers > 1 are combined."""
        def fake_generate(abstract, page_text, prev_page, page_num, src, tgt, fmt, previous_translated):
            return f"ok {page_num}"

        opts = OutputOptions(progressive_save=True)
        with patch.object(svc, "generate_text", side_effect=fake_generate):
            svc._translate_pages_parallel(
                _make_triples(2),
                abstract_text="",
                source_language="Chinese",
                target_language="English",
                output_format="console",
                unit_label="page",
                workers=2,
                opts=opts,
            )

        captured = capsys.readouterr()
        assert "progressive-save" in captured.out.lower() or "progressive_save" in captured.out.lower()

    def test_results_still_returned_when_progressive_save_and_workers_gt_1(self, svc):
        """Even with the incompatible combination, translated pages are returned."""
        def fake_generate(abstract, page_text, prev_page, page_num, src, tgt, fmt, previous_translated):
            return f"translated {page_num}"

        opts = OutputOptions(progressive_save=True)
        with patch.object(svc, "generate_text", side_effect=fake_generate):
            results = svc._translate_pages_parallel(
                _make_triples(2),
                abstract_text="",
                source_language="Chinese",
                target_language="English",
                output_format="console",
                unit_label="page",
                workers=2,
                opts=opts,
            )

        assert len(results) == 2


# ---------------------------------------------------------------------------
# Context-length splitting still works inside a parallel worker
# ---------------------------------------------------------------------------

class TestContextLengthSplittingInWorker:

    def test_context_exceeded_splits_and_retries_within_worker(self, svc):
        """When a worker's page triggers CONTEXT_LENGTH_EXCEEDED it splits and retries."""
        from src.services.api_errors import APISignal

        call_log: list[str] = []

        def fake_translate_page(abstract, page_text, prev_page, src, tgt, fmt, previous_translated):
            call_log.append(page_text[:20])
            # First call (long text) exceeds context; subsequent halves succeed
            if len(page_text) > 50:
                return APISignal.CONTEXT_LENGTH_EXCEEDED
            return f"ok({page_text[:10]})"

        long_text = "A" * 200
        all_triples = [(0, long_text, "")]

        with patch.object(svc, "translate_page_text", side_effect=fake_translate_page):
            results = svc._translate_pages_parallel(
                all_triples,
                abstract_text="",
                source_language="Chinese",
                target_language="English",
                output_format="console",
                unit_label="page",
                workers=1,
                opts=OutputOptions(),
            )

        # The long page should have been split — at least 3 calls (1 fail + 2 halves)
        assert len(call_log) >= 3
        assert len(results) == 1
        # Neither half should be an error placeholder
        assert "Translation error" not in results[0]
