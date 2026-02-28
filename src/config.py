"""
Configuration constants for the CJK Translation script.
"""

import json
import logging
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Language mapping
LANGUAGE_MAP: Dict[str, str] = {
    'C': 'Chinese',
    'J': 'Japanese', 
    'K': 'Korean',
    'E': 'English'
}


def extract_page_nums(page_nums_str: Optional[str]) -> Tuple[int, int]:
    """Extract zero-based start/end page indices from a page selection string."""
    if page_nums_str is None:
        return 0, 0

    if '-' in page_nums_str:
        start_page, end_page = map(int, page_nums_str.split('-'))
        return start_page - 1, end_page - 1

    page_num = int(page_nums_str)
    if page_num <= 0:
        raise ValueError(f"{page_nums_str} is not a valid page number.")
    return page_num - 1, page_num - 1


def make_safe_filename(name: str) -> str:
    """Convert a professor name to a safe filename."""
    safe_name = re.sub(r'[^\w\-_\.]', '_', name)
    safe_name = re.sub(r'_+', '_', safe_name)
    safe_name = safe_name.strip('_')
    return safe_name.lower()


def validate_page_nums(value: str) -> str:
    """Validate page numbers input format for CLI arguments."""
    if not re.match(r"^\d+(-\d+)?$", value):
        raise argparse.ArgumentTypeError("Letters, commas, and other symbols not allowed.")
    return value


def parse_language_code(value: str) -> Union[str, Tuple[str, str]]:
    """Parse language code for OCR or translation commands."""
    if len(value) == 1:
        lang_char = value.upper()
        if lang_char not in LANGUAGE_MAP:
            raise argparse.ArgumentTypeError(f"Invalid language code '{lang_char}'. Use C, J, K, or E.")
        return LANGUAGE_MAP[lang_char]

    if len(value) == 2:
        source_char = value[0].upper()
        target_char = value[1].upper()

        if source_char not in LANGUAGE_MAP:
            raise argparse.ArgumentTypeError(f"Invalid source language code '{source_char}'. Use C, J, K, or E.")
        if target_char not in LANGUAGE_MAP:
            raise argparse.ArgumentTypeError(f"Invalid target language code '{target_char}'. Use C, J, K, or E.")
        if source_char == target_char:
            raise argparse.ArgumentTypeError("Source and target languages cannot be the same.")

        return LANGUAGE_MAP[source_char], LANGUAGE_MAP[target_char]

    raise argparse.ArgumentTypeError(
        "Language code must be 1 character (C, J, K, E for OCR) or 2 characters (CE, JK, etc. for translation)"
    )


def load_professor_config() -> Dict[str, Dict[str, str]]:
    """Load professor configuration from environment variables."""
    professors: Dict[str, Dict[str, str]] = {}

    for key, value in os.environ.items():
        if key.startswith('PROF_') and key.endswith('_NAME'):
            prof_id = key[5:-5]
            primary_key_var = f'PROF_{prof_id}_KEY'
            backup_key_var = f'PROF_{prof_id}_BACKUP_KEY'

            if primary_key_var in os.environ:
                safe_name = make_safe_filename(value)
                professors[safe_name] = {
                    'name': value,
                    'primary_key': primary_key_var,
                    'backup_key': backup_key_var,
                    'id': prof_id,
                    'safe_name': safe_name,
                }

    return professors


def get_api_key(professor_name: str) -> Tuple[str, str]:
    """Get API key and resolved professor display name from environment configuration."""
    professors = load_professor_config()

    if professor_name in professors:
        prof_config = professors[professor_name]
    else:
        prof_config = None
        for _, config in professors.items():
            if config['name'].lower() == professor_name.lower():
                prof_config = config
                break

        if prof_config is None:
            available_names = [config['name'] for config in professors.values()]
            available_safe = list(professors.keys())

            if available_names:
                error_msg = (
                    f"Professor '{professor_name}' not found. Available professors:\n"
                    f"Full names: {', '.join(available_names)}\n"
                    f"CLI names: {', '.join(available_safe)}"
                )
            else:
                error_msg = (
                    "No professors configured. Please set up professor configuration in .env file.\n"
                    "Example: PROF_1_NAME=John Doe, PROF_1_KEY=your_api_key"
                )
            raise ValueError(error_msg)

    primary_key = os.getenv(prof_config['primary_key'])
    if primary_key:
        return primary_key, prof_config['name']

    backup_key = os.getenv(prof_config['backup_key'])
    if backup_key:
        print(f"Warning: Using backup API key for {prof_config['name']}")
        return backup_key, prof_config['name']

    raise ValueError(
        f"No API key found for professor '{prof_config['name']}'. "
        f"Please set {prof_config['primary_key']} in your .env file."
    )

# Constants
MODEL_CATALOG_FILE = "model_catalog.json"
DEFAULT_FALLBACK_MODEL = "gpt-4o-mini"

def get_model_catalog_path() -> Path:
    """Get the path to the model catalog file."""
    return Path(__file__).parent / MODEL_CATALOG_FILE

def load_model_catalog() -> Dict[str, Any]:
    """Load model catalog from file with comprehensive validation."""
    catalog_file = get_model_catalog_path()
    
    if not catalog_file.exists():
        error_msg = (
            f"Model catalog file not found at {catalog_file}. "
            "This file is required for the application to function. "
            "Please create the model catalog file with model pricing and capability information."
        )
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(catalog_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in model catalog file {catalog_file}: {e}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate required sections
    if "config" not in config:
        error_msg = f"Model catalog file {catalog_file} missing required 'config' section."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    if "models" not in config:
        error_msg = f"Model catalog file {catalog_file} missing required 'models' section."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    if not config["models"]:
        error_msg = f"Model catalog file {catalog_file} has no models configured."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    return config

def get_available_models() -> List[str]:
    """Get available models from the model catalog."""
    config = load_model_catalog()
    return list(config["models"].keys())

def get_model_pricing(model: str) -> Dict[str, float]:
    """Get pricing for a specific model."""
    config = load_model_catalog()
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
                f"Model '{model}' not found in model catalog and no fallback model '{DEFAULT_FALLBACK_MODEL}' available. "
                f"Available models: {available_models}. "
                f"Please update your model catalog file."
            )
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    return models[model]

def get_pricing_unit() -> int:
    """Get the pricing unit from the model catalog config."""
    config = load_model_catalog()
    return config["config"]["pricing_unit"]

def get_monthly_limit() -> float:
    """Get the monthly spending limit from the model catalog config."""
    config = load_model_catalog()
    return config["config"]["monthly_limit"]

def model_supports_vision(model: str) -> bool:
    """Check if a model supports vision/image processing."""
    config = load_model_catalog()
    models = config["models"]
    
    if model not in models:
        logging.warning(f"Model {model} not found in pricing config. Assuming no vision support.")
        return False
    
    return models[model].get("supports_vision", False)

def get_vision_capable_models() -> List[str]:
    """Get list of models that support vision/image processing."""
    config = load_model_catalog()
    return [model for model, details in config["models"].items() 
            if details.get("supports_vision", False)]

def get_model_system_role(model: str) -> str:
    """Get the appropriate system message role for a model.
    
    Newer reasoning models (e.g. o3-mini, gpt-5) require 'developer' instead of 'system'.
    Defaults to 'system' for all other models.
    """
    config = load_model_catalog()
    models = config["models"]
    return models.get(model, {}).get("system_role", "system")

def model_uses_max_completion_tokens(model: str) -> bool:
    """Check if a model requires 'max_completion_tokens' instead of 'max_tokens'.
    
    Newer reasoning models (e.g. o3-mini, gpt-5) reject 'max_tokens'.
    """
    config = load_model_catalog()
    models = config["models"]
    return models.get(model, {}).get("use_max_completion_tokens", False)

def model_has_fixed_parameters(model: str) -> bool:
    """Check if a model only accepts default sampling parameters.

    Reasoning models (e.g. o3-mini, gpt-5) reject temperature, top_p,
    frequency_penalty, and presence_penalty — only the default values are supported.
    """
    config = load_model_catalog()
    models = config["models"]
    return models.get(model, {}).get("fixed_parameters", False)

def get_model_max_completion_tokens(model: str, default: int) -> int:
    """Get per-model max completion tokens override, falling back to the given default.

    Reasoning models (e.g. gpt-5) consume hidden reasoning tokens from the same
    completion budget, so they need a larger cap than standard models.
    Set 'max_completion_tokens' in model_catalog.json to override the default.
    """
    config = load_model_catalog()
    return config["models"].get(model, {}).get("max_completion_tokens", default)

def resolve_model(
    requested_model: Optional[str] = None,
    *,
    prefer_model: Optional[str] = None,
    require_vision: bool = False,
) -> str:
    """Resolve a model name from config using deterministic fallback rules.

    Resolution order:
    1) requested_model (if provided and valid)
    2) prefer_model (if provided and valid)
    3) DEFAULT_MODEL (if valid)
    4) first available compatible model from pricing config

    Args:
        requested_model: Optional user-specified model
        prefer_model: Optional mode-specific preferred model (e.g., OCR_MODEL)
        require_vision: Whether selected model must support vision

    Returns:
        Resolved model name

    Raises:
        ValueError: If requested model is invalid/incompatible or no compatible model exists
    """
    available_models = get_available_models()
    compatibility_label = "vision-capable" if require_vision else "configured"
    suggestion = (
        "Use --list-models to see which models support vision."
        if require_vision
        else "Use --list-models to see available options."
    )

    def is_compatible(model_name: str) -> bool:
        return model_supports_vision(model_name) if require_vision else True

    def resolve_candidate(model_name: Optional[str]) -> Optional[str]:
        if model_name and model_name in available_models and is_compatible(model_name):
            return model_name
        return None

    if requested_model:
        # 1) requested_model (if provided and valid)
        if requested_model not in available_models:
            raise ValueError(f"Custom model '{requested_model}' is not configured. {suggestion}")
        if not is_compatible(requested_model):
            raise ValueError(
                f"Custom model '{requested_model}' is not {compatibility_label} for this operation. {suggestion}"
            )
        return requested_model

    # 2) prefer_model (if provided and valid)
    # 3) DEFAULT_MODEL (if valid)
    priority_candidates = [candidate for candidate in (prefer_model, DEFAULT_MODEL) if candidate]
    for candidate in priority_candidates:
        resolved = resolve_candidate(candidate)
        if resolved:
            return resolved

    # 4) first available compatible model from pricing config
    for model in available_models:
        if model in priority_candidates:
            continue
        resolved = resolve_candidate(model)
        if resolved:
            return resolved

    raise ValueError(
        f"No {compatibility_label} models available. Available models: {available_models}. "
        "Please update model_catalog.json."
    )

def save_model_catalog(config: Dict[str, Any]) -> None:
    """Save model catalog to file."""
    catalog_file = get_model_catalog_path()
    with open(catalog_file, 'w') as f:
        json.dump(config, f, indent=2)

# Default model
DEFAULT_MODEL: str = "gpt-4o"

# OCR-specific model (can override DEFAULT_MODEL for image processing)
OCR_MODEL: str = "gpt-4o"  # More cost-effective for text extraction

# Combined image transcription + translation model
# Defaults to a reasoning vision model since simultaneous OCR + translation
# benefits from reasoning about ambiguous characters in context.
IMAGE_TRANSLATION_MODEL: str = "gpt-5"
IMAGE_TRANSLATION_MAX_TOKENS: int = 8000  # Overridden per-model via max_completion_tokens in catalog

# Translation parameters
TRANSLATION_TEMPERATURE: float = 0.5
TRANSLATION_MAX_TOKENS: int = 4000  # Increased from 1000 to handle academic content with footnotes
TRANSLATION_TOP_P: float = 0.5

# OCR parameters (more conservative to reduce hallucination)
OCR_TEMPERATURE: float = 0.0  # Deterministic output
OCR_MAX_TOKENS: int = 4000  # Same as translation for long documents
OCR_TOP_P: float = 0.1  # Very low to prevent creativity
OCR_FREQUENCY_PENALTY: float = 0.5  # Penalize repetition of tokens
OCR_PRESENCE_PENALTY: float = 0.3  # Encourage diversity

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
