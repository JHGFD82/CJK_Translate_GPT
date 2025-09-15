"""
Utility functions for the CJK Translation CLI.
"""

import argparse
import os
import re
from typing import Tuple, Dict

from .config import LANGUAGE_MAP


def make_safe_filename(name: str) -> str:
    """Convert a professor name to a safe filename by replacing spaces and special chars with underscores."""
    # Replace spaces and special characters with underscores
    safe_name = re.sub(r'[^\w\-_\.]', '_', name)
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name.lower()


def validate_page_nums(value: str) -> str:
    """Validate the page numbers input."""
    if not re.match(r"^\d+(-\d+)?$", value):
        raise argparse.ArgumentTypeError("Letters, commas, and other symbols not allowed.")
    return value


def parse_language_code(value: str) -> Tuple[str, str]:
    """Parse language code like 'CE' into source and target languages."""
    if len(value) != 2:
        raise argparse.ArgumentTypeError("Language code must be exactly 2 characters (e.g., CE, JK, etc.)")
    
    source_char = value[0].upper()
    target_char = value[1].upper()
    
    if source_char not in LANGUAGE_MAP:
        raise argparse.ArgumentTypeError(f"Invalid source language code '{source_char}'. Use C, J, K, or E.")
    if target_char not in LANGUAGE_MAP:
        raise argparse.ArgumentTypeError(f"Invalid target language code '{target_char}'. Use C, J, K, or E.")
    if source_char == target_char:
        raise argparse.ArgumentTypeError("Source and target languages cannot be the same.")
    
    return LANGUAGE_MAP[source_char], LANGUAGE_MAP[target_char]


def load_professor_config() -> Dict[str, Dict[str, str]]:
    """Load professor configuration from environment variables.
    
    Returns:
        Dict with professor names as keys (safe filename format) and config as values
    """
    professors: Dict[str, Dict[str, str]] = {}
    
    # Look for all PROF_[ID]_NAME variables
    for key, value in os.environ.items():
        if key.startswith('PROF_') and key.endswith('_NAME'):
            # Extract the ID from PROF_[ID]_NAME
            prof_id = key[5:-5]  # Remove 'PROF_' and '_NAME'
            
            # Get the corresponding keys
            primary_key_var = f'PROF_{prof_id}_KEY'
            backup_key_var = f'PROF_{prof_id}_BACKUP_KEY'
            
            # Check if required keys exist
            if primary_key_var in os.environ:
                safe_name = make_safe_filename(value)
                professors[safe_name] = {
                    'name': value,
                    'primary_key': primary_key_var,
                    'backup_key': backup_key_var,
                    'id': prof_id,
                    'safe_name': safe_name
                }
    
    return professors


def get_api_key(professor_name: str) -> Tuple[str, str]:
    """Get API key for the specified professor name from environment variables.
    
    Args:
        professor_name: Professor name or safe filename version
        
    Returns:
        Tuple of (api_key, actual_professor_name)
        
    Raises:
        ValueError: If professor not found or no API key available
    """
    professors = load_professor_config()
    
    # Try direct match first (safe filename format)
    if professor_name in professors:
        prof_config = professors[professor_name]
    else:
        # Try to find by original name (case-insensitive)
        prof_config = None
        for safe_name, config in professors.items():
            if config['name'].lower() == professor_name.lower():
                prof_config = config
                break
        
        if prof_config is None:
            # Generate helpful error message
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
                    f"No professors configured. Please set up professor configuration in .env file.\n"
                    f"Example: PROF_1_NAME=John Doe, PROF_1_KEY=your_api_key"
                )
            raise ValueError(error_msg)
    
    # Try to get the primary API key
    primary_key = os.getenv(prof_config['primary_key'])
    if primary_key:
        return primary_key, prof_config['name']
    
    # Try backup key if primary not available
    backup_key = os.getenv(prof_config['backup_key'])
    if backup_key:
        print(f"Warning: Using backup API key for {prof_config['name']}")
        return backup_key, prof_config['name']
    
    # No keys available
    raise ValueError(
        f"No API key found for professor '{prof_config['name']}'. "
        f"Please set {prof_config['primary_key']} in your .env file."
    )
