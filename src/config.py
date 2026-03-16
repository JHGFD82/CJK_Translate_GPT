"""
Configuration constants for the PU AI Sandbox.
"""

import json
import logging
import argparse
import os
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Language mapping
LANGUAGE_MAP: Dict[str, str] = {
    'C': 'Chinese',
    'S': 'Simplified Chinese',
    'T': 'Traditional Chinese',
    'J': 'Japanese',
    'K': 'Korean',
    'E': 'English',
}


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


def _language_keys_str() -> str:
    """Return a human-readable list of valid language keys derived from LANGUAGE_MAP."""
    return ', '.join(sorted(LANGUAGE_MAP.keys()))


def parse_single_language_code(value: str) -> str:
    """Parse a single language code (e.g. E, C, J, K) for transcribe/OCR commands."""
    lang_char = value.upper()
    if len(lang_char) != 1 or lang_char not in LANGUAGE_MAP:
        raise argparse.ArgumentTypeError(
            f"Invalid language code '{value}'. Use one of: {_language_keys_str()}."
        )
    return LANGUAGE_MAP[lang_char]


def parse_language_code(value: str) -> Union[str, Tuple[str, str]]:
    """Parse language code for OCR or translation commands."""
    valid_keys = _language_keys_str()

    if len(value) == 1:
        lang_char = value.upper()
        if lang_char not in LANGUAGE_MAP:
            raise argparse.ArgumentTypeError(f"Invalid language code '{lang_char}'. Use one of: {valid_keys}.")
        return LANGUAGE_MAP[lang_char]

    if len(value) == 2:
        source_char = value[0].upper()
        target_char = value[1].upper()

        if source_char not in LANGUAGE_MAP:
            raise argparse.ArgumentTypeError(f"Invalid source language code '{source_char}'. Use one of: {valid_keys}.")
        if target_char not in LANGUAGE_MAP:
            raise argparse.ArgumentTypeError(f"Invalid target language code '{target_char}'. Use one of: {valid_keys}.")
        if source_char == target_char:
            raise argparse.ArgumentTypeError("Source and target languages cannot be the same.")

        return LANGUAGE_MAP[source_char], LANGUAGE_MAP[target_char]

    raise argparse.ArgumentTypeError(
        f"Language code must be 1 character ({valid_keys} for OCR) or 2 characters (CE, JK, etc. for translation)"
    )


_PROF_NAME_PATTERN = re.compile(r'^PROF_(.+?)_NAME$')


def load_professor_config() -> Dict[str, Dict[str, str]]:
    """Load professor configuration from environment variables."""
    professors: Dict[str, Dict[str, str]] = {}

    for key, value in os.environ.items():
        match = _PROF_NAME_PATTERN.match(key)
        if match:
            prof_id = match.group(1)
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


LLMPRICES_API_BASE = "https://llmprices.ai/api/pricing"


def maybe_sync_model_pricing(model: str) -> None:
    """Fetch current pricing for a model from llmprices.ai if not yet synced this month.

    Only runs when the model has a 'llmprices_id' field. Silently skips on any
    network or parse error so the caller is never blocked.
    """
    try:
        catalog = load_model_catalog()
        model_entry = catalog["models"].get(model, {})
        llmprices_id = model_entry.get("llmprices_id")
        if not llmprices_id:
            return

        current_month = datetime.now().strftime("%Y-%m")
        if model_entry.get("last_sync") == current_month:
            return

        url = f"{LLMPRICES_API_BASE}?model={llmprices_id}"
        with urllib.request.urlopen(url, timeout=5) as response:  # nosec B310
            data = json.loads(response.read())

        pricing = data.get("pricing", {})
        prompt_price = float(pricing.get("prompt", 0))
        completion_price = float(pricing.get("completion", 0))

        if prompt_price > 0 and completion_price > 0:
            pricing_unit = catalog["config"]["pricing_unit"]
            catalog["models"][model]["input"] = round(prompt_price * pricing_unit, 4)
            catalog["models"][model]["output"] = round(completion_price * pricing_unit, 4)
            catalog["models"][model]["last_sync"] = current_month
            save_model_catalog(catalog)
            logging.info(
                f"Synced pricing for {model} from llmprices.ai: "
                f"input=${catalog['models'][model]['input']}, "
                f"output=${catalog['models'][model]['output']}"
            )
    except Exception as e:
        logging.warning(f"Could not sync pricing for {model} from llmprices.ai: {e}")


def force_sync_all_pricing() -> dict:
    """Force-sync pricing for all models that have a llmprices_id, regardless of last_sync.

    Returns a dict mapping model name → 'updated' | 'unchanged' | 'failed'.
    """
    catalog = load_model_catalog()
    pricing_unit = catalog["config"]["pricing_unit"]
    current_month = datetime.now().strftime("%Y-%m")
    results: dict = {}

    for model, entry in catalog["models"].items():
        llmprices_id = entry.get("llmprices_id")
        if not llmprices_id:
            continue
        try:
            url = f"{LLMPRICES_API_BASE}?model={llmprices_id}"
            with urllib.request.urlopen(url, timeout=5) as response:  # nosec B310
                data = json.loads(response.read())

            pricing = data.get("pricing", {})
            prompt_price = float(pricing.get("prompt", 0))
            completion_price = float(pricing.get("completion", 0))

            if prompt_price > 0 and completion_price > 0:
                new_input = round(prompt_price * pricing_unit, 4)
                new_output = round(completion_price * pricing_unit, 4)
                old_input = entry.get("input")
                old_output = entry.get("output")
                catalog["models"][model]["input"] = new_input
                catalog["models"][model]["output"] = new_output
                catalog["models"][model]["last_sync"] = current_month
                changed = (new_input != old_input or new_output != old_output)
                results[model] = "updated" if changed else "unchanged"
            else:
                results[model] = "unchanged"
        except Exception as e:
            logging.warning(f"Could not sync pricing for {model}: {e}")
            results[model] = "failed"

    save_model_catalog(catalog)
    return results


# Default model used by resolve_model() as the final named fallback
DEFAULT_MODEL: str = "gpt-4o"
