"""Info/report command handlers for CLI runtime actions."""

import argparse
import logging
from typing import Optional

from ..config import load_model_catalog, get_pricing_unit
from ..errors import CLIError
from ..tracking.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


def list_available_models() -> None:
    """List all available models and their capabilities."""
    config = load_model_catalog()
    models = config["models"]
    pricing_unit = get_pricing_unit()

    print("\n=== Available Models ===")
    print(f"Pricing is per {pricing_unit:,} tokens\n")

    for model_name, pricing in sorted(models.items()):
        vision = "✓" if pricing.get("supports_vision", False) else "✗"
        print(f"{model_name}")
        print(f"  Vision Support: {vision}")
        print(f"  Input:  ${pricing['input']:.3f}")
        print(f"  Output: ${pricing['output']:.3f}")
        print()


def _print_daily_usage(token_tracker: TokenTracker, professor_name: str, date: Optional[str] = None) -> None:
    """Display daily usage report for info-only command path."""
    if date == 'today':
        usage = token_tracker.get_daily_usage()
        print(f"\nToday's usage for {professor_name}:")
    else:
        usage = token_tracker.get_daily_usage(date)
        print(f"\nUsage for {date} for {professor_name}:")

    if not usage.get('models'):
        print("No usage recorded for this date.")
        return

    print(f"Total tokens: {usage['total_tokens']:,}")
    print(f"Total cost: ${usage['total_cost']:.4f}")
    print("\nBy model:")
    for model, model_usage in usage['models'].items():
        print(f"  {model}: {model_usage['total_tokens']:,} tokens, ${model_usage['total_cost']:.4f}")


def handle_info_commands(args: argparse.Namespace) -> bool:
    """Handle info/reporting commands without API-key dependent runtime initialization."""
    # Global info commands (no professor required)
    if args.list_models:
        list_available_models()
        return True

    if args.update_pricing:
        model, input_price, output_price = args.update_pricing
        try:
            input_price_float = float(input_price)
            output_price_float = float(output_price)
        except ValueError as e:
            raise CLIError("Prices must be valid numbers") from e

        # Update pricing doesn't need a professor-specific tracker
        token_tracker = TokenTracker(professor="global")
        token_tracker.update_pricing(model, input_price_float, output_price_float)
        logger.info(f"Updated pricing for {model}: Input=${input_price_float}, Output=${output_price_float}")
        print(f"Updated pricing for {model}: Input=${input_price_float}, Output=${output_price_float}")
        return True

    # Usage commands (professor required)
    if getattr(args, 'command', None) == 'usage':
        if not args.professor:
            raise CLIError("Professor name is required for usage commands.")

        token_tracker = TokenTracker(professor=args.professor)
        usage_subcommand = getattr(args, 'usage_subcommand', None)

        if usage_subcommand == 'report':
            token_tracker.print_usage_report()
            return True

        if usage_subcommand == 'daily':
            date = getattr(args, 'date', 'today')
            _print_daily_usage(token_tracker, args.professor, date)
            return True

        raise CLIError("Invalid usage subcommand. Use 'report' or 'daily'.")

    return False
