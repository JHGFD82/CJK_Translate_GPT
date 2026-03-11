"""
Tests for token tracking data structures and math:
  - UsageStats.add_usage, merge_dict, to_dict
  - TokenTracker._update_stats
  - TokenTracker._calculate_costs  (mocked pricing)
  - TokenTracker._get_monthly_budget_status  (mocked usage data)

No API calls, no cloud I/O; disk writes are directed to tmp_path.
"""

from unittest.mock import patch

import pytest

from src.tracking.token_tracker import UsageStats, TokenTracker


# ---------------------------------------------------------------------------
# UsageStats
# ---------------------------------------------------------------------------

class TestUsageStats:

    def test_initial_values_are_zero(self):
        stats = UsageStats()
        assert stats.total_tokens == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_cost == 0.0
        assert stats.call_count == 0

    def test_add_usage_single_call(self):
        stats = UsageStats()
        stats.add_usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.01)
        assert stats.total_tokens == 150
        assert stats.total_input_tokens == 100
        assert stats.total_output_tokens == 50
        assert stats.total_cost == pytest.approx(0.01)
        assert stats.call_count == 1

    def test_add_usage_accumulates_across_calls(self):
        stats = UsageStats()
        stats.add_usage(100, 50, 150, 0.01)
        stats.add_usage(200, 100, 300, 0.02)
        assert stats.total_tokens == 450
        assert stats.total_input_tokens == 300
        assert stats.total_output_tokens == 150
        assert stats.total_cost == pytest.approx(0.03)
        assert stats.call_count == 2

    def test_merge_dict_full(self):
        stats = UsageStats()
        d = {
            "total_tokens": 100,
            "total_input_tokens": 60,
            "total_output_tokens": 40,
            "total_cost": 0.05,
            "call_count": 3,
        }
        stats.merge_dict(d)
        assert stats.total_tokens == 100
        assert stats.total_input_tokens == 60
        assert stats.total_output_tokens == 40
        assert stats.total_cost == pytest.approx(0.05)
        assert stats.call_count == 3

    def test_merge_dict_partial_uses_zero_defaults(self):
        stats = UsageStats()
        stats.merge_dict({"total_tokens": 50})
        assert stats.total_tokens == 50
        assert stats.total_input_tokens == 0
        assert stats.total_cost == 0.0

    def test_merge_dict_accumulates_with_existing(self):
        stats = UsageStats(total_tokens=10, call_count=1)
        stats.merge_dict({"total_tokens": 20, "call_count": 2})
        assert stats.total_tokens == 30
        assert stats.call_count == 3

    def test_to_dict_returns_all_fields(self):
        stats = UsageStats(
            total_tokens=10,
            total_input_tokens=6,
            total_output_tokens=4,
            total_cost=0.01,
            call_count=1,
        )
        d = stats.to_dict()
        assert d == {
            "total_tokens": 10,
            "total_input_tokens": 6,
            "total_output_tokens": 4,
            "total_cost": 0.01,
            "call_count": 1,
        }

    def test_to_dict_roundtrip_via_merge_dict(self):
        original = UsageStats(total_tokens=99, total_input_tokens=60,
                               total_output_tokens=39, total_cost=0.15, call_count=5)
        restored = UsageStats()
        restored.merge_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()


# ---------------------------------------------------------------------------
# Fixtures for TokenTracker tests (uses tmp_path to avoid touching the real data/)
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker(tmp_path):
    """A TokenTracker for a synthetic professor backed by a temp file."""
    data_file = str(tmp_path / "token_usage_test.json")
    with patch("src.tracking.token_tracker.get_monthly_limit", return_value=100.0):
        t = TokenTracker("testprof", data_file=data_file, monthly_limit=100.0)
    return t


# ---------------------------------------------------------------------------
# TokenTracker._update_stats
# ---------------------------------------------------------------------------

class TestUpdateStats:

    def test_increments_all_fields(self, tracker):
        stats = {"total_tokens": 0, "total_input_tokens": 0,
                 "total_output_tokens": 0, "total_cost": 0.0}
        tracker._update_stats(stats, prompt_tokens=10, completion_tokens=5,
                              total_tokens=15, cost=0.02)
        assert stats["total_tokens"] == 15
        assert stats["total_input_tokens"] == 10
        assert stats["total_output_tokens"] == 5
        assert stats["total_cost"] == pytest.approx(0.02)
        assert stats["call_count"] == 1

    def test_accumulates_on_repeated_calls(self, tracker):
        stats = {"total_tokens": 0, "total_input_tokens": 0,
                 "total_output_tokens": 0, "total_cost": 0.0}
        tracker._update_stats(stats, 10, 5, 15, 0.01)
        tracker._update_stats(stats, 20, 10, 30, 0.02)
        assert stats["total_tokens"] == 45
        assert stats["call_count"] == 2

    def test_initialises_call_count_when_missing(self, tracker):
        stats = {"total_tokens": 0, "total_input_tokens": 0,
                 "total_output_tokens": 0, "total_cost": 0.0}
        # call_count key absent — setdefault should create it
        tracker._update_stats(stats, 1, 1, 2, 0.001)
        assert "call_count" in stats
        assert stats["call_count"] == 1


# ---------------------------------------------------------------------------
# TokenTracker._calculate_costs
# ---------------------------------------------------------------------------

class TestCalculateCosts:

    def test_standard_pricing(self, tracker):
        with patch("src.tracking.token_tracker.get_pricing_unit", return_value=1_000_000), \
             patch("src.tracking.token_tracker.get_model_pricing",
                   return_value={"input": 3.0, "output": 10.0}):
            in_cost, out_cost, total = tracker._calculate_costs(
                "gpt-4o", prompt_tokens=1_000_000, completion_tokens=500_000
            )
        assert in_cost == pytest.approx(3.0)    # 1M / 1M * 3.0
        assert out_cost == pytest.approx(5.0)   # 0.5M / 1M * 10.0
        assert total == pytest.approx(8.0)

    def test_zero_tokens_yields_zero_cost(self, tracker):
        with patch("src.tracking.token_tracker.get_pricing_unit", return_value=1_000_000), \
             patch("src.tracking.token_tracker.get_model_pricing",
                   return_value={"input": 3.0, "output": 10.0}):
            in_cost, out_cost, total = tracker._calculate_costs("gpt-4o", 0, 0)
        assert in_cost == 0.0
        assert out_cost == 0.0
        assert total == 0.0

    def test_total_equals_input_plus_output(self, tracker):
        with patch("src.tracking.token_tracker.get_pricing_unit", return_value=1_000_000), \
             patch("src.tracking.token_tracker.get_model_pricing",
                   return_value={"input": 5.0, "output": 15.0}):
            in_cost, out_cost, total = tracker._calculate_costs("gpt-4o", 200_000, 100_000)
        assert total == pytest.approx(in_cost + out_cost)


# ---------------------------------------------------------------------------
# TokenTracker._get_monthly_budget_status
# ---------------------------------------------------------------------------

class TestMonthlyBudgetStatus:

    def _set_total_cost(self, tracker, cost: float):
        """Helper: write a total_cost into the tracker's in-memory usage_data."""
        tracker.usage_data["total_usage"] = UsageStats(total_cost=cost).to_dict()

    def test_under_limit_not_exceeded(self, tracker):
        self._set_total_cost(tracker, 20.0)
        status = tracker._get_monthly_budget_status()
        assert status["is_exceeded"] is False

    def test_over_limit_is_exceeded(self, tracker):
        self._set_total_cost(tracker, 110.0)
        status = tracker._get_monthly_budget_status()
        assert status["is_exceeded"] is True

    def test_usage_percentage_calculated_correctly(self, tracker):
        self._set_total_cost(tracker, 25.0)   # 25/100 = 25%
        status = tracker._get_monthly_budget_status()
        assert status["usage_percentage"] == pytest.approx(25.0)

    def test_remaining_budget_calculated_correctly(self, tracker):
        self._set_total_cost(tracker, 30.0)
        status = tracker._get_monthly_budget_status()
        assert status["remaining_budget"] == pytest.approx(70.0)

    def test_remaining_budget_clamps_to_zero_when_exceeded(self, tracker):
        self._set_total_cost(tracker, 150.0)
        status = tracker._get_monthly_budget_status()
        assert status["remaining_budget"] == 0.0

    def test_approaching_limit_false_below_80_percent(self, tracker):
        self._set_total_cost(tracker, 79.0)
        status = tracker._get_monthly_budget_status()
        assert status["approaching_limit"] is False

    def test_approaching_limit_true_above_80_percent(self, tracker):
        self._set_total_cost(tracker, 81.0)
        status = tracker._get_monthly_budget_status()
        assert status["approaching_limit"] is True

    def test_status_contains_monthly_usage_key(self, tracker):
        self._set_total_cost(tracker, 10.0)
        status = tracker._get_monthly_budget_status()
        assert "monthly_usage" in status
