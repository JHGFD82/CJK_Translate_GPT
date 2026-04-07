"""
Tests for the CLI argument parser (create_argument_parser).

Validates that all subcommands, flags, and type callbacks are correctly wired
without actually invoking any runtime logic or API calls.
"""

import pytest

from src.cli import create_argument_parser


@pytest.fixture
def parser():
    return create_argument_parser()


# ---------------------------------------------------------------------------
# transcribe subcommand
# ---------------------------------------------------------------------------

class TestTranscribeSubcommand:

    def test_command_set_to_transcribe(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "image.jpg"])
        assert args.command == "transcribe"

    def test_language_code_resolved_to_full_name(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "image.jpg"])
        assert args.language_code == "Japanese"

    def test_all_language_codes_resolve(self, parser):
        expected = {
            "E": "English",
            "C": "Chinese",
            "S": "Simplified Chinese",
            "T": "Traditional Chinese",
            "J": "Japanese",
            "K": "Korean",
        }
        for code, name in expected.items():
            args = parser.parse_args(["heller", "transcribe", code, "-i", "img.png"])
            assert args.language_code == name

    def test_input_file_stored(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "scan.png"])
        assert args.input_file == "scan.png"

    def test_output_file_optional(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "scan.png"])
        assert args.output_file is None

    def test_output_file_stored_when_provided(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "scan.png", "-o", "out.txt"])
        assert args.output_file == "out.txt"

    def test_model_flag_optional(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "scan.png"])
        assert args.model is None

    def test_model_flag_stored_when_provided(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "scan.png", "-m", "gpt-4o-mini"])
        assert args.model == "gpt-4o-mini"

    # ----- vertical flag (-v / --vertical) -----

    def test_vertical_flag_defaults_to_false(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "scan.png"])
        assert args.vertical is False

    def test_vertical_short_flag_sets_true(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "scan.png", "-v"])
        assert args.vertical is True

    def test_vertical_long_flag_sets_true(self, parser):
        args = parser.parse_args(["heller", "transcribe", "J", "-i", "scan.png", "--vertical"])
        assert args.vertical is True

    def test_invalid_language_code_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["heller", "transcribe", "X", "-i", "scan.png"])

    def test_missing_input_file_is_none(self, parser):
        # -i is optional at parse time; runtime validation catches the missing file
        args = parser.parse_args(["heller", "transcribe", "J"])
        assert args.input_file is None


# ---------------------------------------------------------------------------
# translate subcommand
# ---------------------------------------------------------------------------

class TestTranslateSubcommand:

    def test_command_set_to_translate(self, parser):
        args = parser.parse_args(["heller", "translate", "JE", "-i", "doc.pdf"])
        assert args.command == "translate"

    def test_language_code_is_tuple(self, parser):
        args = parser.parse_args(["heller", "translate", "JE", "-i", "doc.pdf"])
        assert args.language_code == ("Japanese", "English")

    def test_chinese_to_english(self, parser):
        args = parser.parse_args(["heller", "translate", "CE", "-i", "doc.pdf"])
        assert args.language_code == ("Chinese", "English")

    def test_same_language_pair_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["heller", "translate", "JJ", "-i", "doc.pdf"])

    def test_abstract_flag_defaults_false(self, parser):
        args = parser.parse_args(["heller", "translate", "JE", "-i", "doc.pdf"])
        assert args.abstract is False

    def test_abstract_flag_set(self, parser):
        args = parser.parse_args(["heller", "translate", "JE", "-i", "doc.pdf", "-a"])
        assert args.abstract is True


# ---------------------------------------------------------------------------
# prompt subcommand
# ---------------------------------------------------------------------------

class TestPromptSubcommand:

    def test_command_set_to_prompt(self, parser):
        args = parser.parse_args(["heller", "prompt"])
        assert args.command == "prompt"

    def test_include_system_prompt_defaults_false(self, parser):
        args = parser.parse_args(["heller", "prompt"])
        assert args.include_system_prompt is False

    def test_include_system_prompt_short_flag(self, parser):
        args = parser.parse_args(["heller", "prompt", "-s"])
        assert args.include_system_prompt is True

    def test_include_system_prompt_long_flag(self, parser):
        args = parser.parse_args(["heller", "prompt", "--system"])
        assert args.include_system_prompt is True

    def test_no_user_prompt_argument(self, parser):
        args = parser.parse_args(["heller", "prompt"])
        assert not hasattr(args, "user_prompt")

    def test_output_file_defaults_none(self, parser):
        args = parser.parse_args(["heller", "prompt"])
        assert args.output_file is None

    def test_output_file_stored_when_provided(self, parser):
        args = parser.parse_args(["heller", "prompt", "-o", "out.txt"])
        assert args.output_file == "out.txt"

    def test_model_defaults_none(self, parser):
        args = parser.parse_args(["heller", "prompt"])
        assert args.model is None

    def test_model_stored_when_provided(self, parser):
        args = parser.parse_args(["heller", "prompt", "-m", "gpt-4o"])
        assert args.model == "gpt-4o"

    def test_dry_run_defaults_false(self, parser):
        args = parser.parse_args(["heller", "prompt"])
        assert args.dry_run is False

    def test_dry_run_sets_true(self, parser):
        args = parser.parse_args(["heller", "prompt", "--dry-run"])
        assert args.dry_run is True

    def test_all_flags_together(self, parser):
        args = parser.parse_args(["heller", "prompt", "-s", "-o", "resp.txt", "-m", "gpt-4o-mini"])
        assert args.include_system_prompt is True
        assert args.output_file == "resp.txt"
        assert args.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Global flags (no professor required)
# ---------------------------------------------------------------------------

class TestGlobalFlags:

    def test_list_models_flag(self, parser):
        args = parser.parse_args(["--list-models"])
        assert args.list_models is True

    def test_list_models_defaults_false(self, parser):
        args = parser.parse_args([])
        assert args.list_models is False

    def test_professor_is_optional(self, parser):
        # Should not raise even with no args
        args = parser.parse_args([])
        assert args.professor is None
