"""
Configuration constants for the PU AI Sandbox.

Language helpers, CLI argument types, and professor configuration live here.
Model catalog, pricing, and model resolution are in src/models/.
"""

import argparse
import os
import re
from typing import Dict, Optional, Tuple, Union

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


# ---------------------------------------------------------------------------
# Re-exports — model catalog, pricing, and resolver live in src/models/
# ---------------------------------------------------------------------------
from .models import (  # noqa: E402  (after professor-config code)
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_MODEL,
    MODEL_CATALOG_FILE,
    PORTKEY_PRICING_API_BASE,
    _PORTKEY_PROVIDER_MAP,
    _fetch_model_pricing,
    add_model_to_catalog,
    get_available_models,
    get_model_catalog_path,
    get_model_max_completion_tokens,
    get_model_pricing,
    get_model_system_role,
    get_monthly_limit,
    get_pricing_unit,
    get_vision_capable_models,
    load_model_catalog,
    maybe_sync_model_pricing,
    model_has_fixed_parameters,
    model_supports_vision,
    model_uses_max_completion_tokens,
    resolve_model,
    save_model_catalog,
)

