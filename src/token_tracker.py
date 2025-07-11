"""
Token usage tracking system for the CJK Translation script.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


# Constants
DEFAULT_FALLBACK_MODEL = "gpt-4o-mini"
PRICING_CONFIG_FILE = "pricing_config.json"
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
        self.pricing_config = self._load_pricing_config()
        
        # Set monthly limit from config or parameter
        if monthly_limit is not None:
            self.monthly_limit = monthly_limit
        else:
            self.monthly_limit = self.pricing_config["config"]["monthly_limit"]
        
        self.usage_data = self._load_usage_data()
    
    def _get_pricing_file_path(self) -> Path:
        """Get the path to the pricing configuration file."""
        return Path(__file__).parent / PRICING_CONFIG_FILE
    
    def _save_json_file(self, file_path: Path, data: Dict[str, Any]):
        """Save data to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load data from a JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _load_pricing_config(self) -> Dict[str, Any]:
        """Load pricing configuration from file."""
        pricing_file = self._get_pricing_file_path()
        
        if not pricing_file.exists():
            error_msg = (
                f"Pricing configuration file not found at {pricing_file}. "
                "This file is required for the application to function. "
                "Please create the pricing configuration file with your model pricing information."
            )
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            config = self._load_json_file(pricing_file)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in pricing configuration file {pricing_file}: {e}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate that the config has required structure
        if "config" not in config:
            error_msg = f"Pricing configuration file {pricing_file} missing required 'config' section."
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        if "models" not in config:
            error_msg = f"Pricing configuration file {pricing_file} missing required 'models' section."
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        models = config["models"]
        if not models:
            error_msg = f"Pricing configuration file {pricing_file} has no models configured."
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        return config
    
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
    
    def _get_model_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for a specific model."""
        models = self.pricing_config["models"]
        
        if model not in models:
            # Try to find a fallback model
            if DEFAULT_FALLBACK_MODEL in models:
                logging.warning(f"Model {model} not found in pricing config. Using {DEFAULT_FALLBACK_MODEL} rates.")
                return models[DEFAULT_FALLBACK_MODEL]
            else:
                # No fallback available - this is a configuration error
                available_models = list(models.keys())
                error_msg = (
                    f"Model '{model}' not found in pricing configuration and no fallback model '{DEFAULT_FALLBACK_MODEL}' available. "
                    f"Available models: {available_models}. "
                    f"Please update your pricing configuration file."
                )
                logging.error(error_msg)
                raise ValueError(error_msg)
        
        return models[model]
    
    def _calculate_costs(self, model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float, float]:
        """Calculate costs for input, output, and total tokens."""
        pricing_unit = self.pricing_config["config"]["pricing_unit"]
        pricing = self._get_model_pricing(model)
        
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
        self._update_usage_stats(self.usage_data["total_usage"], prompt_tokens, completion_tokens, total_tokens, total_cost)
        
        # Update model usage
        model_stats = self._get_or_create_usage_stats("model_usage", model)
        self._update_usage_stats(model_stats, prompt_tokens, completion_tokens, total_tokens, total_cost)
        
        # Update daily usage
        date_str = self._get_current_date()
        daily_stats = self._get_or_create_usage_stats("daily_usage", date_str)
        self._update_usage_stats(daily_stats, prompt_tokens, completion_tokens, total_tokens, total_cost)
        
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
    
    def get_remaining_monthly_budget(self, month: Optional[str] = None) -> float:
        """Get remaining budget for the month."""
        monthly_usage = self.get_monthly_usage(month)
        return max(0.0, self.monthly_limit - monthly_usage["total_cost"])
    
    def is_monthly_limit_exceeded(self, month: Optional[str] = None) -> bool:
        """Check if monthly cost limit has been exceeded."""
        monthly_usage = self.get_monthly_usage(month)
        return monthly_usage["total_cost"] >= self.monthly_limit
    
    def get_monthly_usage_percentage(self, month: Optional[str] = None) -> float:
        """Get percentage of monthly limit used."""
        monthly_usage = self.get_monthly_usage(month)
        return (monthly_usage["total_cost"] / self.monthly_limit) * 100 if self.monthly_limit > 0 else 0.0
    
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
        monthly_usage = self.get_monthly_usage()
        usage_percentage = self.get_monthly_usage_percentage()
        remaining_budget = self.get_remaining_monthly_budget()
        
        print(f"\nMonthly Budget Status ({self._get_current_month()}):")
        print("-" * 40)
        print(f"Monthly Limit: ${self.monthly_limit:.2f}")
        print(f"Used: ${monthly_usage['total_cost']:.4f} ({usage_percentage:.1f}%)")
        print(f"Remaining: ${remaining_budget:.2f}")
        
        if self.is_monthly_limit_exceeded():
            print("⚠️  MONTHLY LIMIT EXCEEDED!")
        elif usage_percentage > 80:
            print("⚠️  Approaching monthly limit!")
        
        print("="*60)
    
    def update_pricing(self, model: str, input_price: float, output_price: float):
        """Update pricing for a specific model."""
        # Ensure models section exists
        if "models" not in self.pricing_config:
            self.pricing_config["models"] = {}
        
        self.pricing_config["models"][model] = {"input": input_price, "output": output_price}
        
        # Save updated pricing config
        self._save_json_file(self._get_pricing_file_path(), self.pricing_config)
        
        logging.info(f"Updated pricing for {model}: input=${input_price}, output=${output_price}")
    
    def update_pricing_unit(self, new_unit: int):
        """Update the pricing unit (e.g., change from per 1M tokens to per 1K tokens)."""
        if "config" not in self.pricing_config:
            self.pricing_config["config"] = {}
        
        self.pricing_config["config"]["pricing_unit"] = new_unit
        
        # Save updated pricing config
        self._save_json_file(self._get_pricing_file_path(), self.pricing_config)
        
        logging.info(f"Updated pricing unit to {new_unit:,} tokens")
