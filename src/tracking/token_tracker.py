"""
Token usage tracking system for the CJK Translation script.

Storage layout
--------------
data/
  token_usage_{safe_name}.json          ← current-month active file
  archives/
    {safe_name}/
      2026-02.json                       ← one file per past month
      2026-03.json
      ...

Each JSON file (active or archive) is self-contained for that month:
  {
    "month": "2026-03",
    "total_usage": { ... },    ← totals for THAT month only
    "model_usage":  { ... },
    "daily_usage":  { ... },
    "session_history": [ ... ]
  }

All-time totals are computed on demand by aggregating the current file
plus every file in the archives folder.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from ..config import (
    load_model_catalog, get_model_pricing, get_pricing_unit,
    get_monthly_limit, save_model_catalog
)


# Constants
USAGE_DATA_DIR = "data"
ARCHIVES_SUBDIR = "archives"
USAGE_DATA_FILE = "token_usage.json"  # Legacy root-level file (migration only)


def get_usage_data_path(professor: Optional[str] = None) -> Path:
    """Return the active (current-month) data file path for a professor."""
    project_root = Path(__file__).parent.parent.parent
    base_dir = project_root / USAGE_DATA_DIR
    base_dir.mkdir(exist_ok=True)

    if professor:
        return base_dir / f"token_usage_{professor.lower()}.json"
    else:
        # Legacy path for backward compatibility (no professor specified)
        return project_root / USAGE_DATA_FILE


def get_archive_dir(professor: str) -> Path:
    """Return (and create if needed) the archive directory for a professor."""
    project_root = Path(__file__).parent.parent.parent
    archive_dir = project_root / USAGE_DATA_DIR / ARCHIVES_SUBDIR / professor.lower()
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def get_archive_path(professor: str, month: str) -> Path:
    """Return the archive file path for a professor and month string (e.g. '2026-02')."""
    return get_archive_dir(professor) / f"{month}.json"


@dataclass
class TokenUsage:
    """Token usage data for a single API call."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timestamp: str
    input_cost: float
    output_cost: float
    total_cost: float


@dataclass
class UsageStats:
    """Usage statistics structure."""
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0

    def add_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int, cost: float):
        """Add usage data to the statistics."""
        self.total_tokens += total_tokens
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        self.total_cost += cost
        self.call_count += 1

    def merge_dict(self, d: Dict[str, Any]):
        """Merge a stats dictionary into this object."""
        self.total_tokens += d.get("total_tokens", 0)
        self.total_input_tokens += d.get("total_input_tokens", 0)
        self.total_output_tokens += d.get("total_output_tokens", 0)
        self.total_cost += d.get("total_cost", 0.0)
        self.call_count += d.get("call_count", 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class TokenTracker:
    """Tracks and manages token usage and costs for a specific professor.

    Each active file covers a single calendar month.  When a new month
    begins the previous file is automatically moved to the archives folder
    (``data/archives/{professor}/{YYYY-MM}.json``) and a fresh file is
    started.  All-time totals are computed on demand by aggregating the
    current file with every archive file.
    """

    def __init__(self, professor: Optional[str] = None, data_file: Optional[str] = None,
                 monthly_limit: Optional[float] = None):
        """Initialize the token tracker.

        Args:
            professor:     Professor name used for file naming and archive paths.
            data_file:     Override the default data file path entirely.
            monthly_limit: Override the configured monthly spending limit.
        """
        self.professor = professor

        if data_file:
            self.data_file = Path(data_file)
        else:
            self.data_file = get_usage_data_path(professor)

        self.monthly_limit = monthly_limit if monthly_limit is not None else get_monthly_limit()

        self.usage_data = self._load_usage_data()

        if professor:
            logging.info(f"Token tracking initialized for Professor {professor.title()}: {self.data_file}")
        else:
            logging.info(f"Token tracking initialized (legacy mode): {self.data_file}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_current_date() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _get_current_month() -> str:
        return datetime.now().strftime("%Y-%m")

    def _empty_usage_data(self) -> Dict[str, Any]:
        """Return a fresh, empty monthly data structure stamped with the current month."""
        return {
            "month": self._get_current_month(),
            "total_usage": UsageStats().to_dict(),
            "model_usage": {},
            "daily_usage": {},
            "session_history": [],
        }

    def _build_month_data_from_sessions(self, month: str, sessions: List[Dict],
                                        all_daily: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct a self-contained monthly data blob from a list of session records."""
        total = UsageStats()
        model_usage: Dict[str, UsageStats] = {}

        for s in sessions:
            pt, ct, tt, cost = (s["prompt_tokens"], s["completion_tokens"],
                                s["total_tokens"], s["total_cost"])
            total.add_usage(pt, ct, tt, cost)
            m = s["model"]
            model_usage.setdefault(m, UsageStats()).add_usage(pt, ct, tt, cost)

        daily = {date: usage for date, usage in all_daily.items() if date.startswith(month)}

        return {
            "month": month,
            "total_usage": total.to_dict(),
            "model_usage": {m: s.to_dict() for m, s in model_usage.items()},
            "daily_usage": daily,
            "session_history": sessions,
        }

    def _archive_month(self, data: Dict[str, Any], month: str) -> None:
        """Write *data* to the archive file for *month*, skipping if already archived."""
        if not self.professor:
            return
        archive_path = get_archive_path(self.professor, month)
        if archive_path.exists():
            logging.info(f"Archive already exists for {month}, skipping: {archive_path}")
            return
        with open(archive_path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Archived {self.professor} month {month} → {archive_path}")

    def _migrate_legacy_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Split a legacy (no 'month' key) monolithic file into per-month archives.

        Returns the data for the current month only (or an empty structure if the
        current month has no sessions yet).
        """
        current_month = self._get_current_month()

        # Group raw session records by month prefix of their timestamp
        sessions_by_month: Dict[str, List[Dict]] = {}
        for session in data.get("session_history", []):
            m = session.get("timestamp", "")[:7]  # "YYYY-MM"
            if m:
                sessions_by_month.setdefault(m, []).append(session)

        all_daily = data.get("daily_usage", {})

        # Archive every past month
        for month, sessions in sorted(sessions_by_month.items()):
            if month < current_month:
                month_data = self._build_month_data_from_sessions(month, sessions, all_daily)
                self._archive_month(month_data, month)

        # Return current month's slice (or empty)
        current_sessions = sessions_by_month.get(current_month, [])
        if current_sessions:
            return self._build_month_data_from_sessions(current_month, current_sessions, all_daily)
        return self._empty_usage_data()

    def _load_usage_data(self) -> Dict[str, Any]:
        """Load usage data, handling legacy migration and month rollover."""
        if not self.data_file.exists():
            # Check for old root-level legacy file
            if self.professor:
                legacy_file = get_usage_data_path(None)
                if legacy_file.exists():
                    logging.info(f"Migrating root-level legacy file for {self.professor}")
                    with open(legacy_file, "r") as f:
                        raw = json.load(f)
                    data = self._migrate_legacy_format(raw)
                    self._save_usage_data_to(data)
                    return data
            return self._empty_usage_data()

        with open(self.data_file, "r") as f:
            data = json.load(f)

        # Migrate: old format had no "month" field
        if "month" not in data:
            logging.info(f"Migrating legacy monthly format for {self.professor}")
            data = self._migrate_legacy_format(data)
            self._save_usage_data_to(data)
            return data

        # Rollover: file belongs to a past month → archive it and start fresh
        stored_month = data.get("month", "")
        current_month = self._get_current_month()
        if stored_month < current_month:
            logging.info(f"Month rollover detected for {self.professor}: {stored_month} → {current_month}")
            self._archive_month(data, stored_month)
            fresh = self._empty_usage_data()
            self._save_usage_data_to(fresh)
            return fresh

        return data

    def _save_usage_data_to(self, data: Dict[str, Any]) -> None:
        """Write *data* to self.data_file."""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    def _save_usage_data(self) -> None:
        """Save the current in-memory usage data."""
        self._save_usage_data_to(self.usage_data)

    def _update_stats(self, stats: Dict[str, Any], prompt_tokens: int, completion_tokens: int,
                      total_tokens: int, cost: float) -> None:
        """Mutate a stats dictionary in-place."""
        stats["total_tokens"] += total_tokens
        stats["total_input_tokens"] += prompt_tokens
        stats["total_output_tokens"] += completion_tokens
        stats["total_cost"] += cost
        stats.setdefault("call_count", 0)
        stats["call_count"] += 1

    def _calculate_costs(self, model: str, prompt_tokens: int,
                         completion_tokens: int) -> tuple[float, float, float]:
        """Return (input_cost, output_cost, total_cost) for the given token counts."""
        pricing_unit = get_pricing_unit()
        pricing = get_model_pricing(model)
        input_cost = (prompt_tokens / pricing_unit) * pricing["input"]
        output_cost = (completion_tokens / pricing_unit) * pricing["output"]
        return input_cost, output_cost, input_cost + output_cost

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int,
                     total_tokens: int, requested_model: Optional[str] = None) -> TokenUsage:
        """Record token usage for a single API call.

        Args:
            model:           Actual model name returned by the API (may carry a date suffix).
            prompt_tokens:   Input token count.
            completion_tokens: Output token count.
            total_tokens:    Combined token count.
            requested_model: Model name used in the request (for pricing lookup when different).
        """
        timestamp = datetime.now().isoformat()
        pricing_model = requested_model if requested_model else model
        if requested_model and requested_model != model:
            logging.info(f"Using requested model '{requested_model}' for pricing instead of API model '{model}'")

        input_cost, output_cost, total_cost = self._calculate_costs(pricing_model, prompt_tokens, completion_tokens)

        usage = TokenUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            timestamp=timestamp,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
        )

        self._update_stats(self.usage_data["total_usage"], prompt_tokens, completion_tokens, total_tokens, total_cost)

        if model not in self.usage_data["model_usage"]:
            self.usage_data["model_usage"][model] = UsageStats().to_dict()
        self._update_stats(self.usage_data["model_usage"][model], prompt_tokens, completion_tokens, total_tokens, total_cost)

        date_str = self._get_current_date()
        if date_str not in self.usage_data["daily_usage"]:
            self.usage_data["daily_usage"][date_str] = UsageStats().to_dict()
        self._update_stats(self.usage_data["daily_usage"][date_str], prompt_tokens, completion_tokens, total_tokens, total_cost)

        self.usage_data["session_history"].append(asdict(usage))
        self._save_usage_data()

        logging.info(f"Token usage recorded: {total_tokens} tokens (${total_cost:.4f}) for model {model}")
        return usage

    def get_daily_usage(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Return usage stats for *date* (default: today) from the current month's file."""
        if date is None:
            date = self._get_current_date()
        return self.usage_data["daily_usage"].get(date, UsageStats().to_dict())

    def get_monthly_usage(self, month: Optional[str] = None) -> Dict[str, Any]:
        """Return usage stats for *month* (default: current month).

        For the current month the in-memory totals are returned directly.
        For past months the corresponding archive file is read.
        """
        if month is None:
            month = self._get_current_month()

        if month == self._get_current_month():
            return self.usage_data["total_usage"]

        # Load from archive when requesting a past month
        if self.professor:
            archive_path = get_archive_path(self.professor, month)
            if archive_path.exists():
                with open(archive_path, "r") as f:
                    archive = json.load(f)
                return archive.get("total_usage", UsageStats().to_dict())

        return UsageStats().to_dict()

    def get_all_time_usage(self) -> Dict[str, Any]:
        """Aggregate total usage across all archived months plus the current month."""
        combined = UsageStats()
        combined.merge_dict(self.usage_data["total_usage"])

        if self.professor:
            archive_dir = get_archive_dir(self.professor)
            for archive_file in sorted(archive_dir.glob("*.json")):
                try:
                    with open(archive_file, "r") as f:
                        arc = json.load(f)
                    combined.merge_dict(arc.get("total_usage", {}))
                except (json.JSONDecodeError, KeyError) as e:
                    logging.warning(f"Could not read archive {archive_file}: {e}")

        return combined.to_dict()

    def list_archived_months(self) -> List[str]:
        """Return a sorted list of month strings that have been archived."""
        if not self.professor:
            return []
        archive_dir = get_archive_dir(self.professor)
        return sorted(p.stem for p in archive_dir.glob("*.json"))

    def _get_monthly_budget_status(self, month: Optional[str] = None) -> Dict[str, Any]:
        """Return a dict summarising budget consumption for *month*."""
        monthly_usage = self.get_monthly_usage(month)
        usage_pct = (monthly_usage["total_cost"] / self.monthly_limit) * 100 if self.monthly_limit > 0 else 0.0
        remaining = max(0.0, self.monthly_limit - monthly_usage["total_cost"])
        return {
            "monthly_usage": monthly_usage,
            "usage_percentage": usage_pct,
            "remaining_budget": remaining,
            "is_exceeded": monthly_usage["total_cost"] >= self.monthly_limit,
            "approaching_limit": usage_pct > 80,
        }

    def print_usage_report(self, include_all_time: bool = False):
        """Print a formatted usage report for the current month.

        Args:
            include_all_time: When True, also print all-time totals aggregated
                              from all archived months plus the current month.
        """
        current_month = self._get_current_month()
        monthly_total = self.usage_data["total_usage"]

        print("\n" + "=" * 60)
        if self.professor:
            print(f"TOKEN USAGE REPORT - PROFESSOR {self.professor.upper()}")
        else:
            print("TOKEN USAGE REPORT")
        print("=" * 60)

        # ── Current month ──────────────────────────────────────────
        print(f"\nCurrent Month ({current_month}):")
        print("-" * 40)
        print(f"Total Tokens Used: {monthly_total['total_tokens']:,}")
        print(f"  • Input Tokens:  {monthly_total['total_input_tokens']:,}")
        print(f"  • Output Tokens: {monthly_total['total_output_tokens']:,}")
        print(f"Total Cost: ${monthly_total['total_cost']:.4f}")

        print("\nModel Breakdown (this month):")
        print("-" * 40)
        for model, data in self.usage_data["model_usage"].items():
            print(f"{model}:")
            print(f"  • Calls:  {data['call_count']}")
            print(f"  • Tokens: {data['total_tokens']:,}")
            print(f"  • Cost:   ${data['total_cost']:.4f}")

        # Today's usage
        today_usage = self.get_daily_usage()
        if today_usage["total_tokens"] > 0:
            print(f"\nToday's Usage ({self._get_current_date()}):")
            print("-" * 40)
            print(f"Tokens: {today_usage['total_tokens']:,}")
            print(f"Cost:   ${today_usage['total_cost']:.4f}")

        # Monthly budget
        budget_status = self._get_monthly_budget_status()
        print(f"\nMonthly Budget ({current_month}):")
        print("-" * 40)
        print(f"Monthly Limit: ${self.monthly_limit:.2f}")
        print(f"Used:          ${monthly_total['total_cost']:.4f} ({budget_status['usage_percentage']:.1f}%)")
        print(f"Remaining:     ${budget_status['remaining_budget']:.2f}")

        if budget_status["is_exceeded"]:
            print("⚠️  MONTHLY LIMIT EXCEEDED!")
        elif budget_status["approaching_limit"]:
            print("⚠️  Approaching monthly limit!")

        # ── All-time totals (optional) ─────────────────────────────
        if include_all_time:
            archived = self.list_archived_months()
            if archived:
                all_time = self.get_all_time_usage()
                print(f"\nAll-Time Totals (across {len(archived)} archived month(s) + current):")
                print("-" * 40)
                print(f"Total Tokens: {all_time['total_tokens']:,}")
                print(f"  • Input:    {all_time['total_input_tokens']:,}")
                print(f"  • Output:   {all_time['total_output_tokens']:,}")
                print(f"Total Cost:   ${all_time['total_cost']:.4f}")
                print(f"Total Calls:  {all_time['call_count']}")
                print(f"Archived months: {', '.join(archived)}")

        print("=" * 60)

    def update_pricing(self, model: str, input_price: float, output_price: float):
        """Update pricing for a specific model in the model catalog."""
        config = load_model_catalog()
        config.setdefault("models", {})[model] = {"input": input_price, "output": output_price}
        save_model_catalog(config)
        logging.info(f"Updated pricing for {model}: input=${input_price}, output=${output_price}")

