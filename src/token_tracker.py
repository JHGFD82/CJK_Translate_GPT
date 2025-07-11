"""
Token usage tracking system for the CJK Translation script.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from .config import (
    load_pricing_config, get_model_pricing, get_pricing_unit, 
    get_monthly_limit, save_pricing_config
)


# Constants
USAGE_DATA_FILE = "token_usage.json"


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class TokenTracker:
    """Tracks and manages token usage and costs."""
    
    def __init__(self, data_file: Optional[str] = None, monthly_limit: Optional[float] = None):
        """Initialize the token tracker with optional custom data file path and monthly limit."""
        self.data_file = Path(data_file) if data_file else Path(__file__).parent.parent / USAGE_DATA_FILE
        
        # Set monthly limit from config or parameter
        if monthly_limit is not None:
            self.monthly_limit = monthly_limit
        else:
            self.monthly_limit = get_monthly_limit()
        
        self.usage_data = self._load_usage_data()
    
    def _save_json_file(self, file_path: Path, data: Dict[str, Any]):
        """Save data to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load data from a JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _get_default_usage_structure(self) -> Dict[str, Any]:
        """Get the default usage data structure."""
        return {
            "total_usage": UsageStats().to_dict(),
            "model_usage": {},
            "daily_usage": {},
            "session_history": []
        }
    
    def _load_usage_data(self) -> Dict[str, Any]:
        """Load existing usage data from file."""
        if not self.data_file.exists():
            return self._get_default_usage_structure()
        
        return self._load_json_file(self.data_file)
    
    def _save_usage_data(self):
        """Save usage data to file."""
        self._save_json_file(self.data_file, self.usage_data)
    
    def _calculate_costs(self, model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float, float]:
        """Calculate costs for input, output, and total tokens."""
        pricing_unit = get_pricing_unit()
        pricing = get_model_pricing(model)
        
        # Calculate costs using the configurable pricing unit
        input_cost = (prompt_tokens / pricing_unit) * pricing["input"]
        output_cost = (completion_tokens / pricing_unit) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return input_cost, output_cost, total_cost
    
    def _get_or_create_usage_stats(self, category: str, key: str) -> Dict[str, Any]:
        """Get or create usage statistics for a category and key."""
        if key not in self.usage_data[category]:
            self.usage_data[category][key] = UsageStats().to_dict()
        return self.usage_data[category][key]
    
    def _update_usage_stats(self, stats: Dict[str, Any], prompt_tokens: int, completion_tokens: int, 
                          total_tokens: int, cost: float):
        """Update usage statistics."""
        stats["total_tokens"] += total_tokens
        stats["total_input_tokens"] += prompt_tokens
        stats["total_output_tokens"] += completion_tokens
        stats["total_cost"] += cost
        if "call_count" in stats:
            stats["call_count"] += 1
    
    def _update_usage_category(self, category: str, key: str, prompt_tokens: int, completion_tokens: int, 
                             total_tokens: int, cost: float):
        """Update usage statistics for a specific category and key."""
        stats = self._get_or_create_usage_stats(category, key)
        self._update_usage_stats(stats, prompt_tokens, completion_tokens, total_tokens, cost)
    
    def _update_all_usage_categories(self, model: str, prompt_tokens: int, completion_tokens: int, 
                                   total_tokens: int, cost: float):
        """Update usage statistics for all categories (total, model, daily)."""
        # Update total usage
        self._update_usage_stats(self.usage_data["total_usage"], prompt_tokens, completion_tokens, total_tokens, cost)
        
        # Update model usage
        self._update_usage_category("model_usage", model, prompt_tokens, completion_tokens, total_tokens, cost)
        
        # Update daily usage
        date_str = self._get_current_date()
        self._update_usage_category("daily_usage", date_str, prompt_tokens, completion_tokens, total_tokens, cost)
    
    def _get_current_date(self) -> str:
        """Get current date as string."""
        return datetime.now().strftime("%Y-%m-%d")
    
    def _get_current_month(self) -> str:
        """Get current month as string."""
        return datetime.now().strftime("%Y-%m")
    
    def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int, 
                    total_tokens: int, requested_model: Optional[str] = None) -> TokenUsage:
        """Record token usage for a single API call.
        
        Args:
            model: The actual model name returned by the API (may have date suffix)
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            total_tokens: Total number of tokens
            requested_model: The model name we requested (for pricing lookup)
        """
        timestamp = datetime.now().isoformat()
        
        # Use requested_model for pricing if provided, otherwise use the API model name
        pricing_model = requested_model if requested_model else model
        if requested_model and requested_model != model:
            logging.info(f"Using requested model '{requested_model}' for pricing instead of API model '{model}'")
        
        input_cost, output_cost, total_cost = self._calculate_costs(pricing_model, prompt_tokens, completion_tokens)
        
        # Create usage record
        usage = TokenUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            timestamp=timestamp,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
        
        # Update all usage categories
        self._update_all_usage_categories(model, prompt_tokens, completion_tokens, total_tokens, total_cost)
        
        # Add to session history
        self.usage_data["session_history"].append(asdict(usage))
        
        # Save updated data
        self._save_usage_data()
        
        logging.info(f"Token usage recorded: {total_tokens} tokens (${total_cost:.4f}) for model {model}")
        return usage
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of token usage."""
        return {
            "total_usage": self.usage_data["total_usage"],
            "model_breakdown": self.usage_data["model_usage"],
            "recent_sessions": self.usage_data["session_history"][-10:]  # Last 10 sessions
        }
    
    def get_daily_usage(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get usage for a specific date or today."""
        if date is None:
            date = self._get_current_date()
        
        return self.usage_data["daily_usage"].get(date, UsageStats().to_dict())
    
    def get_monthly_usage(self, month: Optional[str] = None) -> Dict[str, Any]:
        """Get usage for a specific month or current month, calculated from daily usage."""
        if month is None:
            month = self._get_current_month()
        
        # Calculate monthly usage from daily usage
        monthly_stats = UsageStats()
        
        for date, usage in self.usage_data["daily_usage"].items():
            if date.startswith(month):
                monthly_stats.add_usage(
                    usage["total_input_tokens"],
                    usage["total_output_tokens"],
                    usage["total_tokens"],
                    usage["total_cost"]
                )
        
        return monthly_stats.to_dict()
    
    def _get_monthly_budget_status(self, month: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive monthly budget status information."""
        monthly_usage = self.get_monthly_usage(month)
        usage_percentage = (monthly_usage["total_cost"] / self.monthly_limit) * 100 if self.monthly_limit > 0 else 0.0
        remaining_budget = max(0.0, self.monthly_limit - monthly_usage["total_cost"])
        is_exceeded = monthly_usage["total_cost"] >= self.monthly_limit
        
        return {
            "monthly_usage": monthly_usage,
            "usage_percentage": usage_percentage,
            "remaining_budget": remaining_budget,
            "is_exceeded": is_exceeded,
            "approaching_limit": usage_percentage > 80
        }
    
    def get_remaining_monthly_budget(self, month: Optional[str] = None) -> float:
        """Get remaining budget for the month."""
        return self._get_monthly_budget_status(month)["remaining_budget"]
    
    def is_monthly_limit_exceeded(self, month: Optional[str] = None) -> bool:
        """Check if monthly cost limit has been exceeded."""
        return self._get_monthly_budget_status(month)["is_exceeded"]
    
    def get_monthly_usage_percentage(self, month: Optional[str] = None) -> float:
        """Get percentage of monthly limit used."""
        return self._get_monthly_budget_status(month)["usage_percentage"]
    
    def print_usage_report(self):
        """Print a formatted usage report."""
        total = self.usage_data["total_usage"]
        
        print("\n" + "="*60)
        print("TOKEN USAGE REPORT")
        print("="*60)
        print(f"Total Tokens Used: {total['total_tokens']:,}")
        print(f"  • Input Tokens: {total['total_input_tokens']:,}")
        print(f"  • Output Tokens: {total['total_output_tokens']:,}")
        print(f"Total Cost: ${total['total_cost']:.4f}")
        
        print("\nModel Breakdown:")
        print("-" * 40)
        for model, data in self.usage_data["model_usage"].items():
            print(f"{model}:")
            print(f"  • Calls: {data['call_count']}")
            print(f"  • Tokens: {data['total_tokens']:,}")
            print(f"  • Cost: ${data['total_cost']:.4f}")
        
        # Today's usage
        today_usage = self.get_daily_usage()
        if today_usage["total_tokens"] > 0:
            print(f"\nToday's Usage ({self._get_current_date()}):")
            print("-" * 40)
            print(f"Tokens: {today_usage['total_tokens']:,}")
            print(f"Cost: ${today_usage['total_cost']:.4f}")
        
        # Monthly usage and budget
        budget_status = self._get_monthly_budget_status()
        monthly_usage = budget_status["monthly_usage"]
        
        print(f"\nMonthly Budget Status ({self._get_current_month()}):")
        print("-" * 40)
        print(f"Monthly Limit: ${self.monthly_limit:.2f}")
        print(f"Used: ${monthly_usage['total_cost']:.4f} ({budget_status['usage_percentage']:.1f}%)")
        print(f"Remaining: ${budget_status['remaining_budget']:.2f}")
        
        if budget_status["is_exceeded"]:
            print("⚠️  MONTHLY LIMIT EXCEEDED!")
        elif budget_status["approaching_limit"]:
            print("⚠️  Approaching monthly limit!")
        
        print("="*60)
    
    def update_pricing(self, model: str, input_price: float, output_price: float):
        """Update pricing for a specific model."""
        # Load current config
        config = load_pricing_config()
        
        # Ensure models section exists
        if "models" not in config:
            config["models"] = {}
        
        config["models"][model] = {"input": input_price, "output": output_price}
        
        # Save updated config
        save_pricing_config(config)
        logging.info(f"Updated pricing for {model}: input=${input_price}, output=${output_price}")
    
    def update_pricing_unit(self, new_unit: int):
        """Update the pricing unit (e.g., change from per 1M tokens to per 1K tokens)."""
        # Load current config
        config = load_pricing_config()
        
        # Ensure config section exists
        if "config" not in config:
            config["config"] = {}
        
        config["config"]["pricing_unit"] = new_unit
        
        # Save updated config
        save_pricing_config(config)
        logging.info(f"Updated pricing unit to {new_unit:,} tokens")
