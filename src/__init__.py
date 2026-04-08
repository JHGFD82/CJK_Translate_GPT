"""PU AI Sandbox: CJK/English document translation and OCR tools for Princeton University."""

from .cli import main
from .runtime import SandboxProcessor
from .services.translation_service import TranslationService
from .output.file_output import FileOutputHandler
from .processors.pdf_processor import PDFProcessor
from .processors.docx_processor import DocxProcessor
from .processors.txt_processor import TxtProcessor
from .processors.base_text_processor import BaseTextProcessor
from .tracking.token_tracker import TokenTracker
from .errors import CLIError
from .config import (
    parse_language_code,
    load_professor_config,
    get_api_key,
    make_safe_filename,
    validate_page_nums
)

__version__ = "3.0.0"
__author__ = "Jeff Heller"
__email__ = "jsheller@princeton.edu"

__all__ = [
    # Main entry points
    "main",
    "SandboxProcessor",
    "CLIError",
    
    # Core services
    "TranslationService",
    "TokenTracker",
    
    # File processing
    "FileOutputHandler",
    "PDFProcessor",
    "DocxProcessor", 
    "TxtProcessor",
    "BaseTextProcessor",
    
    # Utilities
    "parse_language_code",
    "load_professor_config",
    "get_api_key",
    "make_safe_filename",
    "validate_page_nums"
]
