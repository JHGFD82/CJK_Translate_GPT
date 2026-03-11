"""
Tests for configuration utilities:
  - make_safe_filename
  - parse_single_language_code
  - parse_language_code
  - validate_page_nums
"""

import argparse

import pytest

from src.config import (
    make_safe_filename,
    parse_language_code,
    parse_single_language_code,
    validate_page_nums,
)


# ---------------------------------------------------------------------------
# make_safe_filename
# ---------------------------------------------------------------------------

class TestMakeSafeFilename:

    def test_spaces_become_underscores(self):
        assert make_safe_filename("Jeff Heller") == "jeff_heller"

    def test_already_safe_unchanged(self):
        assert make_safe_filename("heller") == "heller"

    def test_uppercase_lowercased(self):
        assert make_safe_filename("HELLER") == "heller"

    def test_hyphens_preserved(self):
        assert make_safe_filename("smith-jones") == "smith-jones"

    def test_dots_preserved(self):
        assert make_safe_filename("prof.heller") == "prof.heller"

    def test_apostrophe_replaced(self):
        assert make_safe_filename("O'Brien") == "o_brien"

    def test_consecutive_spaces_collapse_to_single_underscore(self):
        # Two spaces → two underscores → collapsed to one
        assert make_safe_filename("a  b") == "a_b"

    def test_leading_special_char_stripped(self):
        assert make_safe_filename("!heller") == "heller"

    def test_trailing_special_char_stripped(self):
        assert make_safe_filename("heller!") == "heller"

    def test_empty_string(self):
        assert make_safe_filename("") == ""

    def test_mixed_name_with_dot_and_hyphen(self):
        # "Prof. Smith-Jones": dot + space + hyphen
        assert make_safe_filename("Prof. Smith-Jones") == "prof._smith-jones"

    def test_underscore_passthrough(self):
        # Underscores in the original name are preserved
        assert make_safe_filename("hello_world") == "hello_world"

    def test_numbers_preserved(self):
        assert make_safe_filename("prof2") == "prof2"


# ---------------------------------------------------------------------------
# parse_single_language_code  (used by the transcribe subcommand)
# ---------------------------------------------------------------------------

class TestParseSingleLanguageCode:

    @pytest.mark.parametrize("code,expected", [
        ("E", "English"),
        ("C", "Chinese"),
        ("S", "Simplified Chinese"),
        ("T", "Traditional Chinese"),
        ("J", "Japanese"),
        ("K", "Korean"),
    ])
    def test_valid_uppercase_codes(self, code, expected):
        assert parse_single_language_code(code) == expected

    def test_lowercase_accepted(self):
        assert parse_single_language_code("j") == "Japanese"

    def test_invalid_code_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_single_language_code("X")

    def test_two_char_code_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_single_language_code("JE")

    def test_empty_string_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_single_language_code("")


# ---------------------------------------------------------------------------
# parse_language_code  (single char → OCR, two chars → translation pair)
# ---------------------------------------------------------------------------

class TestParseLanguageCode:

    # --- single character (OCR mode) ---

    @pytest.mark.parametrize("code,expected", [
        ("J", "Japanese"),
        ("E", "English"),
        ("C", "Chinese"),
        ("S", "Simplified Chinese"),
        ("T", "Traditional Chinese"),
        ("K", "Korean"),
    ])
    def test_single_code_ocr_mode(self, code, expected):
        assert parse_language_code(code) == expected

    def test_single_lowercase_accepted(self):
        assert parse_language_code("j") == "Japanese"

    def test_single_invalid_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_language_code("X")

    # --- two characters (translation pair) ---

    @pytest.mark.parametrize("code,expected", [
        ("CE", ("Chinese", "English")),
        ("JE", ("Japanese", "English")),
        ("KE", ("Korean", "English")),
        ("EJ", ("English", "Japanese")),
        ("ST", ("Simplified Chinese", "Traditional Chinese")),
    ])
    def test_translation_pairs(self, code, expected):
        assert parse_language_code(code) == expected

    def test_translation_pair_lowercase(self):
        assert parse_language_code("ce") == ("Chinese", "English")

    def test_translation_pair_mixed_case(self):
        assert parse_language_code("Ce") == ("Chinese", "English")

    def test_same_source_and_target_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_language_code("JJ")

    def test_invalid_source_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_language_code("XE")

    def test_invalid_target_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_language_code("JX")

    def test_three_char_code_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_language_code("JEK")

    def test_empty_string_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_language_code("")


# ---------------------------------------------------------------------------
# validate_page_nums
# ---------------------------------------------------------------------------

class TestValidatePageNums:

    def test_single_page_number(self):
        assert validate_page_nums("5") == "5"

    def test_page_range(self):
        assert validate_page_nums("1-10") == "1-10"

    def test_first_page(self):
        assert validate_page_nums("1") == "1"

    def test_letters_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            validate_page_nums("abc")

    def test_comma_separated_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            validate_page_nums("1,2")

    def test_double_range_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            validate_page_nums("1-2-3")

    def test_empty_string_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            validate_page_nums("")

    def test_space_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            validate_page_nums("1 2")
