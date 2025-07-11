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
    
    def __init__(self, data_file: Optional[str] = None):
        """Initialize the token tracker with optional custom data file path."""
        if data_file:
            self.data_file = Path(data_file)
        else:
            # Default to storing in the project root
            project_root = Path(__file__).parent.parent
            self.data_file = project_root / "token_usage.json"
        
        self.pricing_config = self._load_pricing_config()
        self.usage_data = self._load_usage_data()
    
    def _load_pricing_config(self) -> Dict[str, Dict[str, float]]:
        """Load pricing configuration from file."""
        pricing_file = Path(__file__).parent / "pricing_config.json"
        
        if not pricing_file.exists():
            # Create default pricing config if it doesn't exist
            default_pricing = {
                "o3-mini": {"input": 1.21, "output": 4.84},
                "gpt-4o-mini": {"input": 0.165, "output": 0.66},
                "gpt-4o": {"input": 2.75, "output": 11.00},
                "gpt-35-turbo-16k": {"input": 3.00, "output": 4.00},
                "Mistral-Small": {"input": 1.00, "output": 3.00},
                "Meta-Llama-3-1-8B-Instruct": {"input": 3.000, "output": 0.61},
                "Meta-Llama-3-1-70B-Instruct": {"input": 2.68, "output": 3.54}
            }
            
            with open(pricing_file, 'w') as f:
                json.dump(default_pricing, f, indent=2)
            
            logging.info(f"Created default pricing configuration at {pricing_file}")
            return default_pricing
        
        with open(pricing_file, 'r') as f:
            return json.load(f)
    
    def _load_usage_data(self) -> Dict[str, Any]:
        """Load existing usage data from file."""
        if not self.data_file.exists():
            return {
                "total_usage": {
                    "total_tokens": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost": 0.0
                },
                "model_usage": {},
                "daily_usage": {},
                "session_history": []
            }
        
        with open(self.data_file, 'r') as f:
            return json.load(f)
    
    def _save_usage_data(self):
        """Save usage data to file."""
        with open(self.data_file, 'w') as f:
            json.dump(self.usage_data, f, indent=2)
    
    def _calculate_costs(self, model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float, float]:
        """Calculate costs for input, output, and total tokens."""
        if model not in self.pricing_config:
            logging.warning(f"Model {model} not found in pricing config. Using default rates.")
            # Use gpt-4o-mini as fallback
            pricing = self.pricing_config.get("gpt-4o-mini", {"input": 0.165, "output": 0.66})
        else:
            pricing = self.pricing_config[model]
        
        # Costs are per 1 million tokens
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return input_cost, output_cost, total_cost
    
    def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int, 
                    total_tokens: int) -> TokenUsage:
        """Record token usage for a single API call."""
        timestamp = datetime.now().isoformat()
        input_cost, output_cost, total_cost = self._calculate_costs(model, prompt_tokens, completion_tokens)
        
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
        
        # Update total usage
        self.usage_data["total_usage"]["total_tokens"] += total_tokens
        self.usage_data["total_usage"]["total_input_tokens"] += prompt_tokens
        self.usage_data["total_usage"]["total_output_tokens"] += completion_tokens
        self.usage_data["total_usage"]["total_cost"] += total_cost
        
        # Update model usage
        if model not in self.usage_data["model_usage"]:
            self.usage_data["model_usage"][model] = {
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "call_count": 0
            }
        
        model_data = self.usage_data["model_usage"][model]
        model_data["total_tokens"] += total_tokens
        model_data["total_input_tokens"] += prompt_tokens
        model_data["total_output_tokens"] += completion_tokens
        model_data["total_cost"] += total_cost
        model_data["call_count"] += 1
        
        # Update daily usage
        date_str = datetime.now().strftime("%Y-%m-%d")
        if date_str not in self.usage_data["daily_usage"]:
            self.usage_data["daily_usage"][date_str] = {
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0
            }
        
        daily_data = self.usage_data["daily_usage"][date_str]
        daily_data["total_tokens"] += total_tokens
        daily_data["total_input_tokens"] += prompt_tokens
        daily_data["total_output_tokens"] += completion_tokens
        daily_data["total_cost"] += total_cost
        
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
            date = datetime.now().strftime("%Y-%m-%d")
        
        return self.usage_data["daily_usage"].get(date, {
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0
        })
    
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
        today = datetime.now().strftime("%Y-%m-%d")
        today_usage = self.get_daily_usage(today)
        if today_usage["total_tokens"] > 0:
            print(f"\nToday's Usage ({today}):")
            print("-" * 40)
            print(f"Tokens: {today_usage['total_tokens']:,}")
            print(f"Cost: ${today_usage['total_cost']:.4f}")
        
        print("="*60)
    
    def update_pricing(self, model: str, input_price: float, output_price: float):
        """Update pricing for a specific model."""
        self.pricing_config[model] = {"input": input_price, "output": output_price}
        
        # Save updated pricing config
        pricing_file = Path(__file__).parent / "pricing_config.json"
        with open(pricing_file, 'w') as f:
            json.dump(self.pricing_config, f, indent=2)
        
        logging.info(f"Updated pricing for {model}: input=${input_price}, output=${output_price}")
