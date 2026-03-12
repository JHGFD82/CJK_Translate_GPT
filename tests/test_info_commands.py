"""
Tests for src/runtime/info_commands.py:
  - list_available_models
  - _print_daily_usage
  - handle_info_commands  (all branches)

No API calls, no cloud I/O; TokenTracker is either mocked or backed by tmp_path.
"""

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

import src.runtime.info_commands as info_mod
from src.errors import CLIError
from src.runtime.info_commands import (
    _print_daily_usage,
    handle_info_commands,
    list_available_models,
)

# ---------------------------------------------------------------------------
# Shared test catalog (matches model_catalog.json schema)
# ---------------------------------------------------------------------------

SAMPLE_CATALOG = {
    "config": {
        "pricing_unit": 1_000_000,
        "monthly_limit": 250.0,
    },
    "models": {
        "gpt-4o": {
            "input": 2.75,
            "output": 11.0,
            "supports_vision": True,
        },
        "text-only-model": {
            "input": 0.10,
            "output": 0.30,
            "supports_vision": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ns(**kwargs) -> argparse.Namespace:
    """Build a minimal Namespace, setting sensible falsy defaults for every
    attribute that handle_info_commands inspects."""
    defaults = dict(
        list_models=False,
        update_pricing=None,
        command=None,
        professor=None,
        usage_subcommand=None,
        month=None,
        all_time=False,
        date="today",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# list_available_models
# ---------------------------------------------------------------------------


class TestListAvailableModels:

    def test_prints_model_names(self, capsys, monkeypatch):
        monkeypatch.setattr(info_mod, "load_model_catalog", lambda: SAMPLE_CATALOG)
        monkeypatch.setattr(info_mod, "get_pricing_unit", lambda: 1_000_000)
        list_available_models()
        out = capsys.readouterr().out
        assert "gpt-4o" in out
        assert "text-only-model" in out

    def test_vision_checkmark_shown(self, capsys, monkeypatch):
        monkeypatch.setattr(info_mod, "load_model_catalog", lambda: SAMPLE_CATALOG)
        monkeypatch.setattr(info_mod, "get_pricing_unit", lambda: 1_000_000)
        list_available_models()
        out = capsys.readouterr().out
        assert "✓" in out   # gpt-4o supports vision
        assert "✗" in out   # text-only-model does not

    def test_prints_input_price(self, capsys, monkeypatch):
        monkeypatch.setattr(info_mod, "load_model_catalog", lambda: SAMPLE_CATALOG)
        monkeypatch.setattr(info_mod, "get_pricing_unit", lambda: 1_000_000)
        list_available_models()
        out = capsys.readouterr().out
        assert "2.750" in out

    def test_prints_pricing_unit(self, capsys, monkeypatch):
        monkeypatch.setattr(info_mod, "load_model_catalog", lambda: SAMPLE_CATALOG)
        monkeypatch.setattr(info_mod, "get_pricing_unit", lambda: 1_000_000)
        list_available_models()
        out = capsys.readouterr().out
        assert "1,000,000" in out


# ---------------------------------------------------------------------------
# _print_daily_usage
# ---------------------------------------------------------------------------


class TestPrintDailyUsage:

    def _make_tracker(self, daily_data: dict) -> MagicMock:
        tracker = MagicMock()
        tracker.get_daily_usage.return_value = daily_data
        return tracker

    def test_no_usage_prints_no_usage_message(self, capsys):
        tracker = self._make_tracker({"call_count": 0, "total_tokens": 0,
                                      "total_cost": 0.0})
        _print_daily_usage(tracker, "testprof", date="2026-03-01")
        out = capsys.readouterr().out
        assert "No usage" in out

    def test_usage_present_prints_token_counts(self, capsys):
        tracker = self._make_tracker({
            "call_count": 3,
            "total_tokens": 1500,
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
            "total_cost": 0.05,
        })
        _print_daily_usage(tracker, "testprof", date="2026-03-01")
        out = capsys.readouterr().out
        assert "1,500" in out
        assert "0.0500" in out
        assert "3" in out

    def test_today_keyword_calls_get_daily_usage_without_args(self, capsys):
        tracker = self._make_tracker({"call_count": 0, "total_tokens": 0, "total_cost": 0.0})
        _print_daily_usage(tracker, "testprof", date="today")
        tracker.get_daily_usage.assert_called_once_with()

    def test_specific_date_passed_to_tracker(self, capsys):
        tracker = self._make_tracker({"call_count": 0, "total_tokens": 0, "total_cost": 0.0})
        _print_daily_usage(tracker, "testprof", date="2026-01-15")
        tracker.get_daily_usage.assert_called_once_with("2026-01-15")

    def test_professor_name_in_output(self, capsys):
        tracker = self._make_tracker({"call_count": 1, "total_tokens": 100,
                                      "total_input_tokens": 60, "total_output_tokens": 40,
                                      "total_cost": 0.01})
        _print_daily_usage(tracker, "Dr. Yamamoto", date="2026-03-01")
        out = capsys.readouterr().out
        assert "Dr. Yamamoto" in out


# ---------------------------------------------------------------------------
# handle_info_commands — global (no professor needed) branches
# ---------------------------------------------------------------------------


class TestHandleInfoCommandsGlobal:

    def test_list_models_returns_true(self, monkeypatch, capsys):
        monkeypatch.setattr(info_mod, "load_model_catalog", lambda: SAMPLE_CATALOG)
        monkeypatch.setattr(info_mod, "get_pricing_unit", lambda: 1_000_000)
        args = _make_ns(list_models=True)
        assert handle_info_commands(args) is True

    def test_list_models_prints_output(self, monkeypatch, capsys):
        monkeypatch.setattr(info_mod, "load_model_catalog", lambda: SAMPLE_CATALOG)
        monkeypatch.setattr(info_mod, "get_pricing_unit", lambda: 1_000_000)
        handle_info_commands(_make_ns(list_models=True))
        assert "gpt-4o" in capsys.readouterr().out

    def test_update_pricing_returns_true(self, monkeypatch, capsys, tmp_path):
        mock_tracker = MagicMock()
        with patch("src.runtime.info_commands.TokenTracker", return_value=mock_tracker):
            args = _make_ns(update_pricing=["gpt-4o", "1.50", "6.00"])
            result = handle_info_commands(args)
        assert result is True
        mock_tracker.update_pricing.assert_called_once_with("gpt-4o", 1.50, 6.00)

    def test_update_pricing_invalid_price_raises_cli_error(self):
        args = _make_ns(update_pricing=["gpt-4o", "not-a-number", "6.00"])
        with pytest.raises(CLIError, match="valid numbers"):
            handle_info_commands(args)

    def test_no_matching_flag_returns_false(self):
        args = _make_ns()
        assert handle_info_commands(args) is False


# ---------------------------------------------------------------------------
# handle_info_commands — usage subcommand branches
# ---------------------------------------------------------------------------


class TestHandleInfoCommandsUsage:

    def _make_mock_tracker(self):
        tracker = MagicMock()
        tracker.list_archived_months.return_value = ["2026-01", "2026-02"]
        tracker.get_daily_usage.return_value = {
            "call_count": 2,
            "total_tokens": 800,
            "total_input_tokens": 500,
            "total_output_tokens": 300,
            "total_cost": 0.04,
        }
        return tracker

    def test_usage_report_calls_print_usage_report(self):
        mock_tracker = self._make_mock_tracker()
        with patch("src.runtime.info_commands.TokenTracker", return_value=mock_tracker):
            args = _make_ns(command="usage", professor="testprof",
                            usage_subcommand="report", month=None, all_time=False)
            result = handle_info_commands(args)
        assert result is True
        mock_tracker.print_usage_report.assert_called_once_with(month=None, include_all_time=False)

    def test_usage_report_passes_month_arg(self):
        mock_tracker = self._make_mock_tracker()
        with patch("src.runtime.info_commands.TokenTracker", return_value=mock_tracker):
            args = _make_ns(command="usage", professor="testprof",
                            usage_subcommand="report", month="2026-01", all_time=False)
            handle_info_commands(args)
        mock_tracker.print_usage_report.assert_called_once_with(month="2026-01", include_all_time=False)

    def test_usage_report_passes_all_time_flag(self):
        mock_tracker = self._make_mock_tracker()
        with patch("src.runtime.info_commands.TokenTracker", return_value=mock_tracker):
            args = _make_ns(command="usage", professor="testprof",
                            usage_subcommand="report", month=None, all_time=True)
            handle_info_commands(args)
        mock_tracker.print_usage_report.assert_called_once_with(month=None, include_all_time=True)

    def test_usage_months_returns_true(self, capsys):
        mock_tracker = self._make_mock_tracker()
        with patch("src.runtime.info_commands.TokenTracker", return_value=mock_tracker):
            args = _make_ns(command="usage", professor="testprof",
                            usage_subcommand="months")
            result = handle_info_commands(args)
        assert result is True

    def test_usage_months_prints_archived_list(self, capsys):
        mock_tracker = self._make_mock_tracker()
        with patch("src.runtime.info_commands.TokenTracker", return_value=mock_tracker):
            args = _make_ns(command="usage", professor="testprof",
                            usage_subcommand="months")
            handle_info_commands(args)
        out = capsys.readouterr().out
        assert "2026-01" in out
        assert "2026-02" in out

    def test_usage_months_no_archives_message(self, capsys):
        mock_tracker = self._make_mock_tracker()
        mock_tracker.list_archived_months.return_value = []
        with patch("src.runtime.info_commands.TokenTracker", return_value=mock_tracker):
            args = _make_ns(command="usage", professor="testprof",
                            usage_subcommand="months")
            handle_info_commands(args)
        out = capsys.readouterr().out
        assert "No archived months" in out

    def test_usage_daily_returns_true(self, capsys):
        mock_tracker = self._make_mock_tracker()
        with patch("src.runtime.info_commands.TokenTracker", return_value=mock_tracker):
            args = _make_ns(command="usage", professor="testprof",
                            usage_subcommand="daily", date="today")
            result = handle_info_commands(args)
        assert result is True

    def test_usage_daily_calls_get_daily_usage(self, capsys):
        mock_tracker = self._make_mock_tracker()
        with patch("src.runtime.info_commands.TokenTracker", return_value=mock_tracker):
            args = _make_ns(command="usage", professor="testprof",
                            usage_subcommand="daily", date="2026-03-01")
            handle_info_commands(args)
        mock_tracker.get_daily_usage.assert_called_with("2026-03-01")

    def test_usage_missing_professor_raises_cli_error(self):
        args = _make_ns(command="usage", professor=None, usage_subcommand="report")
        with pytest.raises(CLIError, match="Professor name is required"):
            handle_info_commands(args)

    def test_invalid_usage_subcommand_raises_cli_error(self):
        mock_tracker = MagicMock()
        with patch("src.runtime.info_commands.TokenTracker", return_value=mock_tracker):
            args = _make_ns(command="usage", professor="testprof",
                            usage_subcommand="unknown")
            with pytest.raises(CLIError, match="Invalid usage subcommand"):
                handle_info_commands(args)
