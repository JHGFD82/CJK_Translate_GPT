"""
Configuration constants for the CJK Translation script.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Language mapping
LANGUAGE_MAP: Dict[str, str] = {
    'C': 'Chinese',
    'J': 'Japanese', 
    'K': 'Korean',
    'E': 'English'
}

# Constants
PRICING_CONFIG_FILE = "pricing_config.json"
DEFAULT_FALLBACK_MODEL = "gpt-4o-mini"

def get_pricing_config_path() -> Path:
    """Get the path to the pricing configuration file."""
    return Path(__file__).parent / PRICING_CONFIG_FILE

def load_pricing_config() -> Dict[str, Any]:
    """Load pricing configuration from file with comprehensive validation."""
    pricing_file = get_pricing_config_path()
    
    if not pricing_file.exists():
        error_msg = (
            f"Pricing configuration file not found at {pricing_file}. "
            "This file is required for the application to function. "
            "Please create the pricing configuration file with your model pricing information."
        )
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(pricing_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in pricing configuration file {pricing_file}: {e}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate required sections
    if "config" not in config:
        error_msg = f"Pricing configuration file {pricing_file} missing required 'config' section."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    if "models" not in config:
        error_msg = f"Pricing configuration file {pricing_file} missing required 'models' section."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    if not config["models"]:
        error_msg = f"Pricing configuration file {pricing_file} has no models configured."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    return config

def get_available_models() -> List[str]:
    """Get available models from pricing configuration."""
    config = load_pricing_config()
    return list(config["models"].keys())

def get_model_pricing(model: str) -> Dict[str, float]:
    """Get pricing for a specific model."""
    config = load_pricing_config()
    models = config["models"]
    
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

def get_pricing_unit() -> int:
    """Get the pricing unit from configuration."""
    config = load_pricing_config()
    return config["config"]["pricing_unit"]

def get_monthly_limit() -> float:
    """Get the monthly spending limit from configuration."""
    config = load_pricing_config()
    return config["config"]["monthly_limit"]

def save_pricing_config(config: Dict[str, Any]) -> None:
    """Save pricing configuration to file."""
    pricing_file = get_pricing_config_path()
    with open(pricing_file, 'w') as f:
        json.dump(config, f, indent=2)

# Default model
DEFAULT_MODEL: str = "gpt-4o"

# Translation parameters
TRANSLATION_TEMPERATURE: float = 0.5
TRANSLATION_MAX_TOKENS: int = 4000  # Increased from 1000 to handle academic content with footnotes
TRANSLATION_TOP_P: float = 0.5

# Rate limiting and retry configuration
PAGE_DELAY_SECONDS: float = 3.0  # Delay between pages to prevent content filter triggers
MAX_RETRIES: int = 10  # Maximum retries for content filter errors
BASE_RETRY_DELAY: float = 3.0  # Base delay for exponential backoff

# PDF margins (in points)
PDF_MARGINS = {
    'left': 72,
    'right': 72,
    'top': 72,
    'bottom': 18
}

# Context percentage for previous page
CONTEXT_PERCENTAGE: float = 0.65
