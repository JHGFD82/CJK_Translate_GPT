"""Tests for src/runtime/command_runner.py — _CommandMixin methods."""

import argparse
import sys
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest

from src.runtime.command_runner import _CommandMixin
from src.errors import CLIError
from src.models import OutputOptions


# ---------------------------------------------------------------------------
# Minimal concrete implementation so we can instantiate _CommandMixin
# ---------------------------------------------------------------------------

class _FakeMixin(_CommandMixin):
    """Concrete subclass of _CommandMixin with stub processing methods."""

    def __init__(self):
        self.translation_service = MagicMock()
        self.image_translation_service = MagicMock()
        self.image_processor = MagicMock()
        self.image_processor_service = MagicMock()
        self.pdf_processor = MagicMock()
        self.prompt_service = MagicMock()
        self.transcription_review_service = MagicMock()
        self.token_tracker = MagicMock()
        self.file_output = MagicMock()

    def _detect_and_validate_file(self, file_path: str) -> str:
        return "txt"

    def translate_custom_text(self, source_language, target_language, abstract=False, opts=None):
        pass

    def process_image_translation_folder(self, folder_path, source_language, target_language, opts=None, workers=1):
        pass

    def translate_document(self, file_path, source_language, target_language,
                           page_nums=None, abstract=False, opts=None, workers=1):
        pass

    def process_image_folder(self, folder_path, target_language, output_file=None,
                              vertical=False, passes=1, workers=1):
        pass

    def process_image(self, file_path, target_language, output_file=None, vertical=False, passes=1):
        pass

    def process_prompt(self, user_prompt, system_prompt=None, output_file=None):
        pass

    def process_transcription_review(self, text, language, kanbun=False, kanbun_main=False, output_file=None):
        pass


def _make_mixin():
    return _FakeMixin()


# ---------------------------------------------------------------------------
# _collect_multiline
# ---------------------------------------------------------------------------

class TestCollectMultiline:

    def test_collects_lines_until_sentinel(self, monkeypatch):
        inputs = iter(["line one", "line two", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        result = _CommandMixin._collect_multiline("Enter text")
        assert result == "line one\nline two"

    def test_eof_ends_collection(self, monkeypatch):
        call_count = 0

        def raise_eof(*_):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "first line"
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        result = _CommandMixin._collect_multiline("Enter text")
        assert result == "first line"

    def test_empty_input_returns_empty_string(self, monkeypatch):
        inputs = iter(["---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        result = _CommandMixin._collect_multiline("Enter text")
        assert result == ""

    def test_prints_label_before_collecting(self, monkeypatch, capsys):
        inputs = iter(["---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        _CommandMixin._collect_multiline("My Label")
        out = capsys.readouterr().out
        assert "My Label" in out


# ---------------------------------------------------------------------------
# _collect_notes
# ---------------------------------------------------------------------------

class TestCollectNotes:

    def test_system_note_only(self, monkeypatch):
        inputs = iter(["system", "my system note", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        sys_note, usr_note = _CommandMixin._collect_notes()
        assert sys_note == "my system note"
        assert usr_note is None

    def test_user_note_only(self, monkeypatch):
        inputs = iter(["user", "my user note", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        sys_note, usr_note = _CommandMixin._collect_notes()
        assert sys_note is None
        assert usr_note == "my user note"

    def test_both_note(self, monkeypatch):
        inputs = iter(["both", "shared note", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        sys_note, usr_note = _CommandMixin._collect_notes()
        assert sys_note == "shared note"
        assert usr_note == "shared note"

    def test_separate_notes(self, monkeypatch):
        inputs = iter(["separate", "sys text", "---", "usr text", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        sys_note, usr_note = _CommandMixin._collect_notes()
        assert sys_note == "sys text"
        assert usr_note == "usr text"

    def test_eof_returns_none_none(self, monkeypatch):
        def raise_eof(*_):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        sys_note, usr_note = _CommandMixin._collect_notes()
        assert sys_note is None
        assert usr_note is None

    def test_invalid_choice_reprompts(self, monkeypatch):
        inputs = iter(["invalid", "system", "note text", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        sys_note, usr_note = _CommandMixin._collect_notes()
        assert sys_note == "note text"

    def test_shows_prompts_when_provided(self, monkeypatch, capsys):
        inputs = iter(["user", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        _CommandMixin._collect_notes(system_prompt="SYS", user_prompt="USR")
        out = capsys.readouterr().out
        assert "SYS" in out
        assert "USR" in out

    def test_empty_note_returns_none_none(self, monkeypatch):
        inputs = iter(["both", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        sys_note, usr_note = _CommandMixin._collect_notes()
        assert sys_note is None
        assert usr_note is None


# ---------------------------------------------------------------------------
# _dry_run_display
# ---------------------------------------------------------------------------

class TestDryRunDisplay:

    def test_shows_model_and_prompts(self, capsys):
        _CommandMixin._dry_run_display("gpt-4o", "SYSTEM_PROMPT", "USER_PROMPT")
        out = capsys.readouterr().out
        assert "gpt-4o" in out
        assert "SYSTEM_PROMPT" in out
        assert "USER_PROMPT" in out
        assert "DRY RUN" in out

    def test_shows_optional_sampling_params(self, capsys):
        _CommandMixin._dry_run_display(
            "gpt-4o", "sys", "usr",
            temperature=0.5, top_p=0.9, max_tokens=100
        )
        out = capsys.readouterr().out
        assert "0.5" in out
        assert "0.9" in out
        assert "100" in out

    def test_shows_note_when_provided(self, capsys):
        _CommandMixin._dry_run_display("gpt-4o", "sys", "usr", note="Note here")
        out = capsys.readouterr().out
        assert "Note here" in out

    def test_omits_none_sampling_params(self, capsys):
        _CommandMixin._dry_run_display("gpt-4o", "sys", "usr")
        out = capsys.readouterr().out
        assert "Temperature" not in out
        assert "Top-p" not in out
        assert "Max tokens" not in out


# ---------------------------------------------------------------------------
# _resolve_output_path
# ---------------------------------------------------------------------------

class TestResolveOutputPath:

    def test_returns_none_when_no_output_file(self):
        mixin = _make_mixin()
        args = argparse.Namespace(output_file=None, input_file=None)
        assert mixin._resolve_output_path(args) is None

    def test_absolute_output_file_returned_as_is(self, tmp_path):
        mixin = _make_mixin()
        abs_out = str(tmp_path / "out.txt")
        args = argparse.Namespace(output_file=abs_out, input_file=None)
        result = mixin._resolve_output_path(args)
        assert result == abs_out

    def test_relative_output_placed_next_to_input(self, tmp_path):
        mixin = _make_mixin()
        input_file = str(tmp_path / "input.txt")
        args = argparse.Namespace(output_file="out.txt", input_file=input_file)
        result = mixin._resolve_output_path(args)
        assert result == str(tmp_path / "out.txt")

    def test_relative_output_without_input_file_made_absolute(self, tmp_path):
        mixin = _make_mixin()
        args = argparse.Namespace(output_file="out.txt", input_file=None)
        result = mixin._resolve_output_path(args)
        import os
        assert os.path.isabs(result)
        assert result.endswith("out.txt")


# ---------------------------------------------------------------------------
# run() dispatch
# ---------------------------------------------------------------------------

class TestRunDispatch:

    def _make_args(self, command: str, **kwargs):
        return argparse.Namespace(command=command, **kwargs)

    def test_unknown_command_prints_to_stderr_and_exits(self):
        mixin = _make_mixin()
        args = argparse.Namespace(command="bogus")
        with pytest.raises(SystemExit) as exc_info:
            mixin.run(args)
        assert exc_info.value.code == 1

    def test_keyboard_interrupt_exits_130(self, monkeypatch):
        mixin = _make_mixin()

        def raise_interrupt(args):
            raise KeyboardInterrupt

        monkeypatch.setattr(mixin, "_run_translate", raise_interrupt)
        args = argparse.Namespace(command="translate")
        with pytest.raises(SystemExit) as exc_info:
            mixin.run(args)
        assert exc_info.value.code == 130

    def test_unexpected_exception_exits_1(self, monkeypatch):
        mixin = _make_mixin()

        def boom(args):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(mixin, "_run_translate", boom)
        args = argparse.Namespace(command="translate")
        with pytest.raises(SystemExit) as exc_info:
            mixin.run(args)
        assert exc_info.value.code == 1

    def test_cli_error_prints_and_exits_1(self, monkeypatch):
        mixin = _make_mixin()

        def raise_cli(args):
            raise CLIError("bad args")

        monkeypatch.setattr(mixin, "_run_translate", raise_cli)
        args = argparse.Namespace(command="translate")
        with pytest.raises(SystemExit) as exc_info:
            mixin.run(args)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _run_translate
# ---------------------------------------------------------------------------

class TestRunTranslate:

    def _make_translate_args(self, **overrides):
        defaults = {
            "language_code": ("Chinese", "English"),
            "input_file": None,
            "custom_text": False,
            "output_file": None,
            "auto_save": False,
            "page_nums": None,
            "abstract": False,
            "notes": False,
            "note_both": None,
            "note_system": None,
            "note_user": None,
            "kanbun": False,
            "dry_run": False,
            "workers": 1,
            "progressive_save": False,
            "custom_font": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_non_tuple_language_code_raises_cli_error(self):
        mixin = _make_mixin()
        args = self._make_translate_args(language_code="CE")
        with pytest.raises(CLIError, match="language code"):
            mixin._run_translate(args)

    def test_custom_text_delegates_to_translate_custom_text(self, monkeypatch):
        mixin = _make_mixin()
        called = []
        monkeypatch.setattr(mixin, "translate_custom_text", lambda *a, **kw: called.append(True))
        args = self._make_translate_args(custom_text=True)
        mixin._run_translate(args)
        assert called

    def test_document_file_delegates_to_translate_document(self, tmp_path, monkeypatch):
        mixin = _make_mixin()
        f = tmp_path / "doc.txt"
        f.write_text("content")
        called = []
        monkeypatch.setattr(mixin, "translate_document", lambda *a, **kw: called.append(True))
        args = self._make_translate_args(input_file=str(f))
        mixin._run_translate(args)
        assert called

    def test_inline_note_system_sets_service_note(self):
        mixin = _make_mixin()
        mixin.translate_custom_text = MagicMock()
        args = self._make_translate_args(custom_text=True, note_system="My system note")
        mixin._run_translate(args)
        assert mixin.translation_service.system_note == "My system note"

    def test_inline_note_user_sets_service_note(self):
        mixin = _make_mixin()
        mixin.translate_custom_text = MagicMock()
        args = self._make_translate_args(custom_text=True, note_user="My user note")
        mixin._run_translate(args)
        assert mixin.translation_service.user_note == "My user note"

    def test_note_both_sets_both_services(self):
        mixin = _make_mixin()
        mixin.translate_custom_text = MagicMock()
        args = self._make_translate_args(custom_text=True, note_both="shared note")
        mixin._run_translate(args)
        assert mixin.translation_service.system_note == "shared note"
        assert mixin.translation_service.user_note == "shared note"

    def test_kanbun_flag_sets_service_kanbun(self):
        mixin = _make_mixin()
        mixin.translate_custom_text = MagicMock()
        args = self._make_translate_args(custom_text=True, kanbun=True)
        mixin._run_translate(args)
        assert mixin.translation_service.kanbun is True

    def test_no_input_raises_cli_error(self):
        mixin = _make_mixin()
        args = self._make_translate_args(input_file=None, custom_text=False)
        with pytest.raises(CLIError, match="No input specified"):
            mixin._run_translate(args)


# ---------------------------------------------------------------------------
# _run_transcribe
# ---------------------------------------------------------------------------

class TestRunTranscribe:

    def _make_transcribe_args(self, **overrides):
        defaults = {
            "language_code": "English",
            "input_file": None,
            "output_file": None,
            "vertical": False,
            "passes": 1,
            "workers": 1,
            "dry_run": False,
            "notes": False,
            "note_both": None,
            "note_system": None,
            "note_user": None,
            "kanbun": False,
            "kanbun_main": False,
            "model": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_image_file_delegates_to_process_image(self, tmp_path, monkeypatch):
        mixin = _make_mixin()
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"fake-image")
        called = []
        monkeypatch.setattr(mixin, "process_image", lambda *a, **kw: called.append(True))
        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "image")
        args = self._make_transcribe_args(input_file=str(img))
        mixin._run_transcribe(args)
        assert called

    def test_folder_delegates_to_process_image_folder(self, tmp_path, monkeypatch):
        mixin = _make_mixin()
        folder = tmp_path / "scans"
        folder.mkdir()
        called = []
        monkeypatch.setattr(mixin, "process_image_folder", lambda *a, **kw: called.append(True))
        args = self._make_transcribe_args(input_file=str(folder))
        mixin._run_transcribe(args)
        assert called

    def test_no_input_raises_cli_error(self):
        mixin = _make_mixin()
        args = self._make_transcribe_args(input_file=None)
        with pytest.raises(CLIError, match="Input file is required"):
            mixin._run_transcribe(args)

    def test_non_image_file_raises_cli_error(self, tmp_path, monkeypatch):
        mixin = _make_mixin()
        f = tmp_path / "doc.txt"
        f.write_text("content")
        # _FakeMixin._detect_and_validate_file returns 'txt'
        args = self._make_transcribe_args(input_file=str(f))
        with pytest.raises(CLIError, match="image file"):
            mixin._run_transcribe(args)


# ---------------------------------------------------------------------------
# _run_prompt
# ---------------------------------------------------------------------------

class TestRunPrompt:

    def _make_prompt_args(self, **overrides):
        defaults = {
            "output_file": None,
            "include_system_prompt": False,
            "dry_run": False,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_calls_process_prompt_with_collected_text(self, monkeypatch):
        mixin = _make_mixin()
        inputs = iter(["user text here", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        called_with = []
        monkeypatch.setattr(mixin, "process_prompt", lambda *a, **kw: called_with.append((a, kw)))
        args = self._make_prompt_args()
        mixin._run_prompt(args)
        assert called_with
        assert "user text here" in called_with[0][0][0]

    def test_include_system_prompt_flag_collects_system_text(self, monkeypatch):
        mixin = _make_mixin()
        # First collect is system, second is user
        seq = iter(["my system", "---", "my user", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(seq))
        captured = []
        monkeypatch.setattr(mixin, "process_prompt", lambda *a, **kw: captured.append((a, kw)))
        args = self._make_prompt_args(include_system_prompt=True)
        mixin._run_prompt(args)
        assert captured
        # system_prompt passed as second positional arg
        assert "my system" in str(captured[0])

    def test_dry_run_does_not_call_process_prompt(self, monkeypatch):
        mixin = _make_mixin()
        mixin.prompt_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.prompt_service.build_prompts = MagicMock(return_value=("sys", "usr"))
        called = []
        monkeypatch.setattr(mixin, "process_prompt", lambda *a, **kw: called.append(True))
        args = self._make_prompt_args(dry_run=True)
        mixin._run_prompt(args)
        assert not called

    def test_empty_user_prompt_raises_cli_error(self, monkeypatch):
        mixin = _make_mixin()
        inputs = iter(["---"])  # Empty — just the sentinel
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        args = self._make_prompt_args()
        with pytest.raises(CLIError, match="No prompt"):
            mixin._run_prompt(args)


# ---------------------------------------------------------------------------
# _run_transcription_review
# ---------------------------------------------------------------------------

class TestRunTranscriptionReview:

    def _make_review_args(self, **overrides):
        defaults = {
            "language_code": "Japanese",
            "input_file": None,
            "output_file": None,
            "custom_text": False,
            "kanbun": False,
            "kanbun_main": False,
            "notes": False,
            "note_both": None,
            "note_system": None,
            "note_user": None,
            "dry_run": False,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_input_file_delegates_to_process_transcription_review(self, tmp_path, monkeypatch):
        mixin = _make_mixin()
        f = tmp_path / "transcription.txt"
        f.write_text("transcribed text", encoding="utf-8")
        called = []
        monkeypatch.setattr(mixin, "process_transcription_review",
                            lambda *a, **kw: called.append(True))
        args = self._make_review_args(input_file=str(f))
        mixin._run_transcription_review(args)
        assert called

    def test_custom_text_delegates_to_process_transcription_review(self, monkeypatch):
        mixin = _make_mixin()
        inputs = iter(["transcription here", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        called = []
        monkeypatch.setattr(mixin, "process_transcription_review",
                            lambda *a, **kw: called.append(True))
        args = self._make_review_args(custom_text=True)
        mixin._run_transcription_review(args)
        assert called

    def test_no_input_raises_cli_error(self):
        mixin = _make_mixin()
        args = self._make_review_args(input_file=None, custom_text=False)
        with pytest.raises(CLIError):
            mixin._run_transcription_review(args)

    def test_empty_file_raises_cli_error(self, tmp_path):
        mixin = _make_mixin()
        f = tmp_path / "empty.txt"
        f.write_text("")
        args = self._make_review_args(input_file=str(f))
        with pytest.raises(CLIError, match="empty"):
            mixin._run_transcription_review(args)

    def test_missing_input_file_raises_cli_error(self):
        mixin = _make_mixin()
        args = self._make_review_args(input_file="/no/such/file.txt")
        with pytest.raises(CLIError, match="not found"):
            mixin._run_transcription_review(args)

    def test_empty_custom_text_raises_cli_error(self, monkeypatch):
        mixin = _make_mixin()
        inputs = iter(["---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        args = self._make_review_args(custom_text=True)
        with pytest.raises(CLIError, match="No transcription text"):
            mixin._run_transcription_review(args)


# ---------------------------------------------------------------------------
# _run_translate — dry_run paths (lines 222-279)
# ---------------------------------------------------------------------------

class TestRunTranslateDryRun:

    def _make_translate_args(self, **overrides):
        defaults = {
            "language_code": ("Chinese", "English"),
            "input_file": None,
            "custom_text": False,
            "output_file": None,
            "auto_save": False,
            "page_nums": None,
            "abstract": False,
            "notes": False,
            "note_both": None,
            "note_system": None,
            "note_user": None,
            "kanbun": False,
            "dry_run": True,
            "workers": 1,
            "progressive_save": False,
            "custom_font": None,
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _make_dry_mixin(self):
        mixin = _make_mixin()
        mixin.translation_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.translation_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        mixin.image_translation_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.image_translation_service.build_prompts = MagicMock(return_value=("ISYS", "IUSR"))
        return mixin

    def test_dry_run_no_input_shows_placeholder(self, capsys):
        mixin = self._make_dry_mixin()
        args = self._make_translate_args()  # no input_file, no custom_text
        mixin._run_translate(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_dry_run_txt_file_reads_first_page(self, tmp_path, capsys, monkeypatch):
        mixin = self._make_dry_mixin()
        txt = tmp_path / "doc.txt"
        txt.write_text("Hello world content", encoding="utf-8")

        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "txt")
        # Patch TxtProcessor so it returns a predictable page
        with patch("src.runtime.command_runner.TxtProcessor.process_txt_with_pages", return_value=["Page one text"]):
            args = self._make_translate_args(input_file=str(txt))
            mixin._run_translate(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_dry_run_image_file_shows_image_note(self, tmp_path, capsys, monkeypatch):
        mixin = self._make_dry_mixin()
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")

        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "image")
        args = self._make_translate_args(input_file=str(img))
        mixin._run_translate(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "base64" in out or "image" in out.lower()

    def test_dry_run_custom_text_collects_text(self, monkeypatch, capsys):
        mixin = self._make_dry_mixin()
        inputs = iter(["Hello from custom text", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        args = self._make_translate_args(custom_text=True)
        mixin._run_translate(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_dry_run_abstract_flag_collects_abstract(self, monkeypatch, capsys):
        mixin = self._make_dry_mixin()
        inputs = iter(["Abstract text here", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        args = self._make_translate_args(abstract=True)
        mixin._run_translate(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_dry_run_docx_file_reads_first_section(self, tmp_path, capsys, monkeypatch):
        mixin = self._make_dry_mixin()
        docx = tmp_path / "doc.docx"
        docx.write_bytes(b"fake docx")

        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "docx")
        with patch("src.runtime.command_runner.DocxProcessor.process_docx_with_pages",
                   return_value=["Docx page one"]):
            args = self._make_translate_args(input_file=str(docx))
            mixin._run_translate(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out


# ---------------------------------------------------------------------------
# _run_translate — notes flow (lines 182-204)
# ---------------------------------------------------------------------------

class TestRunTranslateNotes:

    def _make_translate_args(self, **overrides):
        defaults = {
            "language_code": ("Chinese", "English"),
            "input_file": None,
            "custom_text": True,
            "output_file": None,
            "auto_save": False,
            "page_nums": None,
            "abstract": False,
            "notes": True,
            "note_both": None,
            "note_system": None,
            "note_user": None,
            "kanbun": False,
            "dry_run": False,
            "workers": 1,
            "progressive_save": False,
            "custom_font": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_notes_true_custom_text_sets_notes(self, monkeypatch):
        mixin = _make_mixin()
        mixin.translation_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        # _collect_notes: "user" → note text
        inputs = iter(["user", "My user note", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        mixin.translate_custom_text = MagicMock()
        args = self._make_translate_args()
        mixin._run_translate(args)
        assert mixin.translation_service.user_note == "My user note"

    def test_notes_true_image_file_uses_image_prompts(self, tmp_path, monkeypatch):
        mixin = _make_mixin()
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")
        mixin.image_processor.is_image_file = MagicMock(return_value=True)
        mixin.image_translation_service.build_prompts = MagicMock(return_value=("ISYS", "IUSR"))
        mixin.translation_service.build_prompts = MagicMock(return_value=("TSYS", "TUSR"))
        inputs = iter(["user", "Image note", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        mixin.translate_document = MagicMock()
        args = self._make_translate_args(custom_text=False, input_file=str(img))
        mixin._run_translate(args)
        mixin.image_translation_service.build_prompts.assert_called_once()

    def test_notes_true_text_file_uses_translation_prompts(self, tmp_path, monkeypatch):
        mixin = _make_mixin()
        f = tmp_path / "doc.txt"
        f.write_text("content")
        mixin.image_processor.is_image_file = MagicMock(return_value=False)
        mixin.translation_service.build_prompts = MagicMock(return_value=("TSYS", "TUSR"))
        inputs = iter(["user", "Text file note", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        mixin.translate_document = MagicMock()
        args = self._make_translate_args(custom_text=False, input_file=str(f))
        mixin._run_translate(args)
        mixin.translation_service.build_prompts.assert_called()


# ---------------------------------------------------------------------------
# _run_translate — folder path (line 298)
# ---------------------------------------------------------------------------

class TestRunTranslateFolderInput:

    def _make_translate_args(self, input_file=None, **overrides):
        defaults = {
            "language_code": ("Chinese", "English"),
            "input_file": input_file,
            "custom_text": False,
            "output_file": None,
            "auto_save": False,
            "page_nums": None,
            "abstract": False,
            "notes": False,
            "note_both": None,
            "note_system": None,
            "note_user": None,
            "kanbun": False,
            "dry_run": False,
            "workers": 1,
            "progressive_save": False,
            "custom_font": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_directory_input_delegates_to_process_image_translation_folder(self, tmp_path, monkeypatch):
        mixin = _make_mixin()
        folder = tmp_path / "images"
        folder.mkdir()
        called = []
        monkeypatch.setattr(mixin, "process_image_translation_folder",
                            lambda *a, **kw: called.append(True))
        args = self._make_translate_args(input_file=str(folder))
        mixin._run_translate(args)
        assert called


# ---------------------------------------------------------------------------
# _run_transcribe — dry_run path (lines 337-348)
# ---------------------------------------------------------------------------

class TestRunTranscribeDryRun:

    def _make_transcribe_args(self, **overrides):
        defaults = {
            "language_code": "English",
            "input_file": None,
            "output_file": None,
            "vertical": False,
            "passes": 1,
            "workers": 1,
            "dry_run": True,
            "notes": False,
            "note_both": None,
            "note_system": None,
            "note_user": None,
            "kanbun": False,
            "kanbun_main": False,
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
            "model": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_dry_run_shows_output(self, capsys):
        mixin = _make_mixin()
        mixin.image_processor_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.image_processor_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        args = self._make_transcribe_args()
        mixin._run_transcribe(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_dry_run_multi_pass_note_in_output(self, capsys):
        mixin = _make_mixin()
        mixin.image_processor_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.image_processor_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        args = self._make_transcribe_args(passes=3)
        mixin._run_transcribe(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "pass" in out.lower() or "3" in out


# ---------------------------------------------------------------------------
# _run_transcription_review — dry_run path (lines 422-431)
# ---------------------------------------------------------------------------

class TestRunTranscriptionReviewDryRun:

    def _make_review_args(self, **overrides):
        defaults = {
            "language_code": "Japanese",
            "input_file": None,
            "output_file": None,
            "custom_text": False,
            "kanbun": False,
            "kanbun_main": False,
            "notes": False,
            "note_both": None,
            "note_system": None,
            "note_user": None,
            "dry_run": True,
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_dry_run_shows_output_without_calling_review(self, capsys):
        mixin = _make_mixin()
        mixin.transcription_review_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.transcription_review_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        called = []
        mixin.process_transcription_review = lambda *a, **kw: called.append(True)
        args = self._make_review_args()
        mixin._run_transcription_review(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert not called


# ---------------------------------------------------------------------------
# Additional branch-coverage tests for previously untested paths
# ---------------------------------------------------------------------------

class TestCollectNotesBranches:
    """Cover the system_prompt-only and user_prompt-only display branches."""

    def test_shows_only_user_prompt_when_system_is_none(self, monkeypatch, capsys):
        """Branch 97->100: system_prompt is None but user_prompt is not None."""
        inputs = iter(["user", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        _CommandMixin._collect_notes(system_prompt=None, user_prompt="USR_PROMPT")
        out = capsys.readouterr().out
        assert "USR_PROMPT" in out
        assert "SYSTEM PROMPT" not in out

    def test_shows_only_system_prompt_when_user_is_none(self, monkeypatch, capsys):
        """Branch 100->103: system_prompt is not None but user_prompt is None."""
        inputs = iter(["system", "note text", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        sys_note, usr_note = _CommandMixin._collect_notes(system_prompt="SYS_PROMPT", user_prompt=None)
        out = capsys.readouterr().out
        assert "SYS_PROMPT" in out
        assert "USER PROMPT" not in out
        assert sys_note == "note text"


class TestRunTranslateExtraEdgeCases:
    """Cover remaining gaps in _run_translate."""

    def _make_translate_args(self, **overrides):
        defaults = {
            "language_code": ("Chinese", "English"),
            "input_file": None,
            "custom_text": False,
            "output_file": None,
            "auto_save": False,
            "page_nums": None,
            "abstract": False,
            "notes": False,
            "note_both": None,
            "note_system": None,
            "note_user": None,
            "kanbun": False,
            "dry_run": False,
            "workers": 1,
            "progressive_save": False,
            "custom_font": None,
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_tuple_length_not_2_raises_cli_error(self):
        """Line 173: len(lang_tuple) != 2 raises CLIError."""
        mixin = _make_mixin()
        args = self._make_translate_args(language_code=("Chinese",))
        with pytest.raises(CLIError, match="language code"):
            mixin._run_translate(args)

    def test_dry_run_pdf_file_reads_first_page(self, tmp_path, capsys, monkeypatch):
        """Lines 241-243: PDF file in dry-run mode."""
        mixin = _make_mixin()
        mixin.translation_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.translation_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "pdf")
        mixin.pdf_processor.process_pdf = MagicMock(return_value=iter([MagicMock()]))
        mixin.pdf_processor.process_page = MagicMock(return_value="PDF page text")
        args = self._make_translate_args(input_file=str(pdf), dry_run=True)
        mixin._run_translate(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_dry_run_pdf_empty_file_uses_placeholder(self, tmp_path, capsys, monkeypatch):
        """Lines 241-243: empty PDF (no pages) falls back to placeholder."""
        mixin = _make_mixin()
        mixin.translation_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.translation_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-1.4 empty")

        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "pdf")
        mixin.pdf_processor.process_pdf = MagicMock(return_value=iter([]))
        mixin.pdf_processor.process_page = MagicMock(return_value="")
        args = self._make_translate_args(input_file=str(pdf), dry_run=True)
        mixin._run_translate(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_dry_run_empty_custom_text_uses_placeholder(self, monkeypatch, capsys):
        """Line 253: empty custom text in dry-run uses placeholder."""
        mixin = _make_mixin()
        mixin.translation_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.translation_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        # Return empty sentinel immediately
        inputs = iter(["---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        args = self._make_translate_args(custom_text=True, dry_run=True)
        mixin._run_translate(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_dry_run_auto_save_sets_txt_output_format(self, capsys, monkeypatch):
        """Lines 268-269: auto_save=True in dry-run sets output_format to 'txt'."""
        mixin = _make_mixin()
        mixin.translation_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.translation_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        args = self._make_translate_args(dry_run=True, auto_save=True)
        mixin._run_translate(args)
        # build_prompts should be called with output_format='txt'
        call_kwargs = mixin.translation_service.build_prompts.call_args
        assert call_kwargs is not None
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args_positional = call_kwargs.args if call_kwargs.args else ()
        # output_format should be 'txt' — check it was passed
        all_args = str(call_kwargs)
        assert "txt" in all_args

    def test_dry_run_unknown_file_type_uses_placeholder(self, tmp_path, capsys, monkeypatch):
        """Line 253: else-branch when file_type is not pdf/docx/txt/image in dry-run."""
        mixin = _make_mixin()
        mixin.translation_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.translation_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        f = tmp_path / "data.csv"
        f.write_text("a,b,c")
        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "csv")
        args = self._make_translate_args(input_file=str(f), dry_run=True)
        mixin._run_translate(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_dry_run_with_output_file_sets_format_from_extension(self, tmp_path, capsys):
        """Lines 268-269: explicit output_file in dry-run sets output_format from extension."""
        mixin = _make_mixin()
        mixin.translation_service._get_model = MagicMock(return_value="gpt-4o")
        mixin.translation_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        out_file = str(tmp_path / "result.docx")
        args = self._make_translate_args(dry_run=True, output_file=out_file)
        mixin._run_translate(args)
        call_kwargs = mixin.translation_service.build_prompts.call_args
        assert call_kwargs is not None
        all_args = str(call_kwargs)
        assert "docx" in all_args


class TestRunTranscribeExtraEdgeCases:
    """Cover remaining gaps in _run_transcribe."""

    def _make_transcribe_args(self, **overrides):
        defaults = {
            "language_code": "English",
            "input_file": None,
            "output_file": None,
            "vertical": False,
            "passes": 1,
            "workers": 1,
            "dry_run": False,
            "notes": False,
            "note_both": None,
            "note_system": None,
            "note_user": None,
            "kanbun": False,
            "kanbun_main": False,
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
            "model": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_inline_note_system_sets_ocr_service_note(self, tmp_path, monkeypatch):
        """Line 337: note_system sets image_processor_service.system_note."""
        mixin = _make_mixin()
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"fake")
        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "image")
        monkeypatch.setattr(mixin, "process_image", MagicMock())
        args = self._make_transcribe_args(input_file=str(img), note_system="my sys note")
        mixin._run_transcribe(args)
        assert mixin.image_processor_service.system_note == "my sys note"

    def test_inline_note_user_sets_ocr_service_note(self, tmp_path, monkeypatch):
        """Line 339: note_user sets image_processor_service.user_note."""
        mixin = _make_mixin()
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"fake")
        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "image")
        monkeypatch.setattr(mixin, "process_image", MagicMock())
        args = self._make_transcribe_args(input_file=str(img), note_user="my usr note")
        mixin._run_transcribe(args)
        assert mixin.image_processor_service.user_note == "my usr note"

    def test_kanbun_flag_sets_kanbun_on_ocr_service(self, tmp_path, monkeypatch):
        """Line 342: kanbun=True sets image_processor_service.kanbun."""
        mixin = _make_mixin()
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"fake")
        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "image")
        monkeypatch.setattr(mixin, "process_image", MagicMock())
        args = self._make_transcribe_args(input_file=str(img), kanbun=True)
        mixin._run_transcribe(args)
        assert mixin.image_processor_service.kanbun is True

    def test_kanbun_main_flag_sets_kanbun_main_on_ocr_service(self, tmp_path, monkeypatch):
        """Line 345: kanbun_main=True sets image_processor_service.kanbun_main."""
        mixin = _make_mixin()
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"fake")
        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "image")
        monkeypatch.setattr(mixin, "process_image", MagicMock())
        args = self._make_transcribe_args(input_file=str(img), kanbun_main=True)
        mixin._run_transcribe(args)
        assert mixin.image_processor_service.kanbun_main is True

    def test_passes_zero_raises_cli_error(self, tmp_path, monkeypatch):
        """Line 370: passes < 1 raises CLIError."""
        mixin = _make_mixin()
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"fake")
        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "image")
        args = self._make_transcribe_args(input_file=str(img), passes=0)
        with pytest.raises(CLIError, match="passes"):
            mixin._run_transcribe(args)

    def test_notes_flag_triggers_collect_notes_for_transcribe(self, tmp_path, monkeypatch, capsys):
        """Lines 324-330: notes=True shows prompts and collects notes."""
        mixin = _make_mixin()
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"fake")
        monkeypatch.setattr(mixin, "_detect_and_validate_file", lambda p: "image")
        monkeypatch.setattr(mixin, "process_image", MagicMock())
        mixin.image_processor_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        inputs = iter(["user", "my note", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        args = self._make_transcribe_args(input_file=str(img), notes=True)
        mixin._run_transcribe(args)
        assert mixin.image_processor_service.user_note == "my note"


class TestRunTranscriptionReviewExtraEdgeCases:
    """Cover remaining gaps in _run_transcription_review."""

    def _make_review_args(self, **overrides):
        defaults = {
            "language_code": "Japanese",
            "input_file": None,
            "output_file": None,
            "custom_text": False,
            "kanbun": False,
            "kanbun_main": False,
            "notes": False,
            "note_both": None,
            "note_system": None,
            "note_user": None,
            "dry_run": False,
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_notes_flag_triggers_collect_notes(self, tmp_path, monkeypatch, capsys):
        """Lines 408-411: notes=True shows prompts and collects notes."""
        mixin = _make_mixin()
        f = tmp_path / "trans.txt"
        f.write_text("transcription content", encoding="utf-8")
        mixin.transcription_review_service.build_prompts = MagicMock(return_value=("SYS", "USR"))
        monkeypatch.setattr(mixin, "process_transcription_review", MagicMock())
        inputs = iter(["user", "my review note", "---"])
        monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
        args = self._make_review_args(input_file=str(f), notes=True)
        mixin._run_transcription_review(args)
        assert mixin.transcription_review_service.user_note == "my review note"

    def test_inline_note_system_sets_review_service_note(self, tmp_path, monkeypatch):
        """Line 417: note_system sets transcription_review_service.system_note."""
        mixin = _make_mixin()
        f = tmp_path / "trans.txt"
        f.write_text("some transcription", encoding="utf-8")
        monkeypatch.setattr(mixin, "process_transcription_review", MagicMock())
        args = self._make_review_args(input_file=str(f), note_system="reviewer sys note")
        mixin._run_transcription_review(args)
        assert mixin.transcription_review_service.system_note == "reviewer sys note"

    def test_inline_note_user_sets_review_service_note(self, tmp_path, monkeypatch):
        """Line 419: note_user sets transcription_review_service.user_note."""
        mixin = _make_mixin()
        f = tmp_path / "trans.txt"
        f.write_text("some transcription", encoding="utf-8")
        monkeypatch.setattr(mixin, "process_transcription_review", MagicMock())
        args = self._make_review_args(input_file=str(f), note_user="reviewer usr note")
        mixin._run_transcription_review(args)
        assert mixin.transcription_review_service.user_note == "reviewer usr note"
