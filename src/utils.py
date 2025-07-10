"""
Utility functions for the CJK Translation script.
"""

import re
import argparse
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime

from .config import LANGUAGE_MAP


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


def extract_page_nums(page_nums_str: Optional[str]) -> Tuple[int, int]:
    """Extract the start and end page numbers from the given string."""
    if page_nums_str is None:
        return 0, 0  # Process all pages
    
    if '-' in page_nums_str:
        start_page, end_page = map(int, page_nums_str.split('-'))
        return start_page - 1, end_page - 1
    else:
        page_num = int(page_nums_str)
        if page_num <= 0:
            raise ValueError(f"{page_nums_str} is not a valid page number.")
        return page_num - 1, page_num - 1


def generate_output_filename(input_file: str, source_lang: str, target_lang: str, extension: str = '.txt') -> str:
    """Generate an output filename based on input file and languages."""
    input_path = Path(input_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{input_path.stem}_{source_lang}to{target_lang}_{timestamp}{extension}"
    return str(input_path.parent / output_name)


def generate_process_text(abstract_text: str, page_text: str, previous_page: str, context_percentage: float = 0.65) -> str:
    """Generate text for processing with context."""
    context = abstract_text if abstract_text else previous_page[int(len(previous_page) * context_percentage):]
    if context:
        context = f"--Context: \n{context}"
    return f"--Current Page: \n{page_text}\n{context}"
