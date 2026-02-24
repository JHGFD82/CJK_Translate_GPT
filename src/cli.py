"""
Command-line interface for the CJK Translation script.
"""

import argparse
import logging
import os
import sys
from typing import Optional, List

from dotenv import load_dotenv

from .utils import parse_language_code, get_api_key, validate_page_nums
from .services.translation_service import TranslationService
from .file_output import FileOutputHandler
from .processors.docx_processor import DocxProcessor
from .processors.txt_processor import TxtProcessor
from .processors.image_processor import ImageProcessor
from .services.image_processor_service import ImageProcessorService
from .token_tracker import TokenTracker

# Load environment variables
load_dotenv()

# Set up module logger
logger = logging.getLogger(__name__)


# Custom exception for CLI errors
class CLIError(Exception):
    """Raised for user-facing CLI errors."""
    pass


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Translate documents between Chinese, Japanese, Korean, and English using PortKey API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Language codes:
  CE = Chinese to English    EC = English to Chinese
  JE = Japanese to English   EJ = English to Japanese  
  KE = Korean to English     EK = English to Korean
  CJ = Chinese to Japanese   JC = Japanese to Chinese
  CK = Chinese to Korean     KC = Korean to Chinese
  JK = Japanese to Korean    KJ = Korean to Japanese

Examples:
  python main.py professor_name CE -i document.pdf
  python main.py professor_name JE -i document.docx -p 1-5
  python main.py professor_name KE -c
        """
    )
    
    # Positional arguments
    parser.add_argument('professor', type=str, nargs='?',
                        help='Professor name for API key lookup (required for translation, OCR, and usage reports)')
    
    parser.add_argument('language_code', type=parse_language_code, nargs='?',
                        help='Translation direction (CE, JE, KE, etc.)')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-i', '--input', dest='input_file', type=str,
                             help='Input file path (PDF, Word document, or text file)')
    
    input_group.add_argument('-c', '--custom', dest='custom_text', action='store_true',
                             help='Input custom text to be translated')

    # Optional arguments
    parser.add_argument('-p', '--page_nums', dest='page_nums', type=validate_page_nums,
                        help='Page numbers to process\\nEnter either a single page number or a range in this format: '
                             '[starting page number]-[ending page number]\\nNo spaces, letters, commas or other symbols '
                             'are allowed. For PDFs: actual page numbers. For Word/text files: logical pages based on content length')

    parser.add_argument('-a', '--abstract', dest='abstract', action='store_true',
                        help='The text has an abstract')

    parser.add_argument('-o', '--output', dest='output_file', type=str,
                        help='Output file path to save the translation (with .txt, .pdf, or .docx extension)')

    parser.add_argument('--auto-save', dest='auto_save', action='store_true',
                        help='Automatically save output to a timestamped file in the same directory as input PDF')

    parser.add_argument('--progressive-save', dest='progressive_save', action='store_true',
                        help='Save each page to output file immediately after translation (useful for error recovery)')

    parser.add_argument('-f', '--font', dest='custom_font', type=str,
                        help='Custom font name to use for PDF and Word document generation (font must be in fonts/ directory)')

    parser.add_argument('-m', '--model', dest='model', type=str,
                        help='Specify which model to use (e.g., gpt-4o, gpt-4o-mini, gpt-5)')

    # Information commands
    parser.add_argument('--list-models', dest='list_models', action='store_true',
                        help='List all available models and their capabilities')

    # Token usage commands
    parser.add_argument('--usage-report', dest='usage_report', action='store_true',
                        help='Display token usage and cost report')
    
    parser.add_argument('--daily-usage', dest='daily_usage', type=str, nargs='?', const='today',
                        help='Display daily usage for a specific date (YYYY-MM-DD format) or today if no date specified')
    
    parser.add_argument('--update-pricing', dest='update_pricing', type=str, nargs=3, metavar=('MODEL', 'INPUT_PRICE', 'OUTPUT_PRICE'),
                        help='Update pricing for a specific model (e.g., --update-pricing gpt-4 0.03 0.06)')

    return parser


def list_available_models() -> None:
    """List all available models and their capabilities without initializing services."""
    from .config import load_model_catalog, get_pricing_unit

    config = load_model_catalog()
    models = config["models"]
    pricing_unit = get_pricing_unit()

    print("\n=== Available Models ===")
    print(f"Pricing is per {pricing_unit:,} tokens\n")

    for model_name, pricing in sorted(models.items()):
        vision = "✓" if pricing.get("supports_vision", False) else "✗"
        print(f"{model_name}")
        print(f"  Vision Support: {vision}")
        print(f"  Input:  ${pricing['input']:.3f} per {pricing_unit:,} tokens")
        print(f"  Output: ${pricing['output']:.3f} per {pricing_unit:,} tokens")
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


def handle_info_commands_without_processor(args: argparse.Namespace) -> bool:
    """Handle info/reporting commands without API-key dependent service initialization."""
    if args.list_models:
        list_available_models()
        return True

    if args.usage_report or args.daily_usage is not None:
        if not args.professor:
            raise CLIError("Professor name is required for usage commands.")

        token_tracker = TokenTracker(professor=args.professor)

        if args.usage_report:
            token_tracker.print_usage_report()
            return True

        if args.daily_usage is not None:
            _print_daily_usage(token_tracker, args.professor, args.daily_usage)
            return True

    if args.update_pricing:
        model, input_price, output_price = args.update_pricing
        try:
            input_price_float = float(input_price)
            output_price_float = float(output_price)
        except ValueError as e:
            raise CLIError("Prices must be valid numbers") from e

        token_tracker = TokenTracker(professor=args.professor)
        token_tracker.update_pricing(model, input_price_float, output_price_float)
        logger.info(f"Updated pricing for {model}: Input=${input_price_float}, Output=${output_price_float}")
        print(f"Updated pricing for {model}: Input=${input_price_float}, Output=${output_price_float}")
        return True

    return False


class SandboxProcessor:
    """Main application class for processing inputs to the Princeton AI Sandbox."""
    
    def __init__(self, professor_name: str, model: Optional[str] = None):
        """Initialize the processor for the specified professor.
        
        Args:
            professor_name: Name of the professor
            model: Optional model name to use instead of defaults
            
        Raises:
            CLIError: If configuration is invalid
        """
        try:
            api_key, self.professor_display_name = get_api_key(professor_name)
            self.professor_name = professor_name
            
            logger.info(f"Initializing processor for professor: {self.professor_display_name}")
            
            # Create shared token tracker for both services
            self.token_tracker = TokenTracker(professor=professor_name)
            
            # Initialize services with shared token tracker and optional model
            self.translation_service = TranslationService(
                api_key, professor_name, token_tracker=self.token_tracker, model=model
            )
            self.image_processor_service = ImageProcessorService(
                api_key, professor_name, token_tracker=self.token_tracker, model=model
            )
            
            self.image_processor = ImageProcessor()
            self.file_output = FileOutputHandler()
            self.setup_logging()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise CLIError(f"Configuration error: {e}") from e
    
    def setup_logging(self) -> None:
        """Set up logging for this session."""
        setup_logging()

    def _detect_and_validate_file(self, file_path: str) -> str:
        """Detect file type and validate the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type as string: 'image', 'pdf', 'docx', 'txt'
            
        Raises:
            CLIError: If file doesn't exist, is invalid, or has unsupported format
        """
        abs_path = os.path.abspath(file_path)
        
        # Check if file exists
        if not os.path.exists(abs_path):
            raise CLIError(f"File '{file_path}' not found.")
        
        logger.debug(f"Validating file: {file_path}")
        
        # Detect file type - check extensions directly
        lower_path = abs_path.lower()
        
        if self.image_processor.is_image_file(abs_path):
            if not self.image_processor.validate_image_file(abs_path):
                raise CLIError(f"Image file '{file_path}' is not valid.")
            logger.info(f"Detected image file: {file_path}")
            return 'image'
        elif lower_path.endswith('.pdf'):
            logger.info(f"Detected PDF file: {file_path}")
            return 'pdf'
        elif lower_path.endswith('.docx'):
            logger.info(f"Detected Word document: {file_path}")
            return 'docx'
        elif lower_path.endswith('.txt'):
            logger.info(f"Detected text file: {file_path}")
            return 'txt'
        else:
            raise CLIError(
                f"Unsupported file format. Supported formats: PDF, DOCX, TXT, or image files (JPG, PNG, etc.)"
            )

    def _handle_page_range(self, pages: List[str], page_nums: Optional[str], file_type: str) -> List[str]:
        """Handle page range selection for text-based documents.
        
        Args:
            pages: List of pages/sections
            page_nums: Page range string (e.g., "1" or "1-3")
            file_type: Type of file being processed (for error messages)
            
        Returns:
            Filtered list of pages based on page range
            
        Raises:
            CLIError: If page range is invalid
        """
        if not page_nums:
            return pages
            
        from .utils import extract_page_nums
        start_page, end_page = extract_page_nums(page_nums)
        
        # Ensure page range is valid
        if start_page >= len(pages):
            raise CLIError(
                f"Page {start_page + 1} does not exist. Document has {len(pages)} logical pages."
            )
        
        # Limit end_page to available pages
        end_page = min(end_page, len(pages) - 1)
        
        # Select the requested page range
        selected_pages = pages[start_page:end_page + 1]
        logger.info(
            f"Processing pages {start_page + 1}-{end_page + 1} of {file_type} "
            f"(logical pages based on content length)"
        )
        
        return selected_pages

    def _process_text_based_file(self, file_path: str, file_type: str, page_nums: Optional[str],
                                 abstract_text: Optional[str], source_language: str, target_language: str,
                                 output_file: Optional[str], auto_save: bool, progressive_save: bool) -> List[str]:
        """Process text-based files (DOCX, TXT) with common logic.
        
        Args:
            file_path: Absolute path to the file
            file_type: Either 'docx' or 'txt'
            page_nums: Page range string (optional)
            abstract_text: Abstract text for context (optional)
            source_language: Source language
            target_language: Target language
            output_file: Output file path (optional)
            auto_save: Whether to auto-save
            progressive_save: Whether to save progressively
            
        Returns:
            List of translated text pages
        """
        logger.info(f"Processing {file_type.upper()} file: {file_path}")
        
        if file_type == 'docx':
            with open(file_path, 'rb') as f:
                pages = DocxProcessor.process_docx_with_pages(f, target_page_size=2000)
                pages = self._handle_page_range(pages, page_nums, "Word document")
        elif file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                pages = TxtProcessor.process_txt_with_pages(f, target_page_size=2000)
                pages = self._handle_page_range(pages, page_nums, "text file")
        else:
            raise ValueError(f"Unsupported text file type: {file_type}")
        
        logger.info(f"Translating {len(pages)} page(s) from {source_language} to {target_language}")
        
        return self.translation_service.translate_text_pages(
            pages, abstract_text, source_language, target_language, 
            output_file, auto_save, progressive_save, file_path
        )

    def translate_document(self, file_path: str, source_language: str, target_language: str,
                          page_nums: Optional[str] = None, abstract: bool = False,
                          output_file: Optional[str] = None, auto_save: bool = False, progressive_save: bool = False, 
                          custom_font: Optional[str] = None) -> None:
        """Translate a document file (PDF, Word document, or text file)."""
        # Convert file_path to absolute path and detect type
        file_path = os.path.abspath(file_path)
        file_type = self._detect_and_validate_file(file_path)
        
        # Get abstract text once if needed
        abstract_text = input('Enter abstract text: ') if abstract else None
        
        logger.info(f"Starting translation: {source_language} → {target_language}")
        
        try:
            # Process based on file type
            if file_type == 'pdf':
                logger.info(f"Processing PDF file: {file_path}")
                with open(file_path, 'rb') as f:
                    pages = self.translation_service.pdf_processor.process_pdf(f)
                    logger.info("Translating PDF pages")
                    document_text = self.translation_service.translate_document(
                        pages, abstract_text, page_nums, source_language, target_language, 
                        output_file, auto_save, progressive_save, file_path
                    )
            elif file_type in ('docx', 'txt'):
                document_text = self._process_text_based_file(
                    file_path, file_type, page_nums, abstract_text, source_language, target_language,
                    output_file, auto_save, progressive_save
                )
            else:
                # This shouldn't happen due to earlier validation, but handle it gracefully
                raise CLIError(f"Cannot translate file type '{file_type}'.")
            
            # Join all translated content
            full_translation = "".join(document_text)
            
            # Display the translation
            print(full_translation)
            
            # Save the translation if requested (skip if progressive saving was used)
            if not progressive_save and (output_file or auto_save):
                logger.info(f"Saving translation output")
                self.file_output.save_translation_output(
                    full_translation, file_path, output_file, auto_save, source_language, target_language, custom_font
                )
            
            logger.info("Translation completed successfully")
                
        except ImportError as e:
            if "python-docx" in str(e):
                logger.error("python-docx library not found")
                raise CLIError(
                    "python-docx is required to process Word documents. "
                    "Install it with: pip install python-docx"
                ) from e
            else:
                logger.error(f"Import error: {e}")
                raise CLIError(f"Import error: {e}") from e
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            raise CLIError(f"Error processing document: {e}") from e
    
    def translate_custom_text(self, source_language: str, target_language: str,
                            output_file: Optional[str] = None, auto_save: bool = False, custom_font: Optional[str] = None) -> None:
        """Translate custom text input by the user."""
        print(f"Enter the {source_language} text you want to translate to {target_language}:")
        print("(Press Ctrl+D on Unix/Linux/Mac or Ctrl+Z followed by Enter on Windows to finish)")
        
        logger.info(f"Starting custom text translation: {source_language} -> {target_language}")
        
        try:
            custom_text = ""
            while True:
                try:
                    line = input()
                    custom_text += line + "\n"
                except EOFError:
                    break
            
            if not custom_text.strip():
                logger.warning("No text provided for translation")
                print("No text provided.")
                return
            
            print("\nTranslating...")
            translated_text = self.translation_service.translate_text(custom_text, source_language, target_language)
            
            # Save the translation if requested
            if output_file or auto_save:
                input_filename = f"custom_text_{source_language}to{target_language}.txt"
                logger.info("Saving custom text translation")
                self.file_output.save_translation_output(
                    translated_text, input_filename, output_file, auto_save, source_language, target_language, custom_font
                )
            
            logger.info("Custom text translation completed successfully")
        except KeyboardInterrupt:
            logger.info("Translation cancelled by user")
            print("\nTranslation cancelled.")
        except Exception as e:
            logger.error(f"Error during translation: {e}", exc_info=True)
            raise CLIError(f"Error during translation: {e}") from e

    def process_image(self, file_path: str, target_language: str, output_file: Optional[str] = None) -> None:
        """Process an image file with OCR.
        
        Raises:
            CLIError: If image processing fails
        """
        logger.info(f"Starting OCR processing: {file_path} → {target_language}")
        
        try:
            # Perform OCR
            extracted_text = self.image_processor_service.process_image_ocr(
                file_path, target_language, output_format="console"
            )
            
            print("\n=== Extracted Text ===")
            print(extracted_text)
            print("======================\n")
            
            # Save to file if output_file is specified
            if output_file:
                output_path = os.path.abspath(output_file)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                logger.info(f"Extracted text saved to: {output_path}")
                print(f"Extracted text saved to: {output_path}")
            
            logger.info("OCR processing completed successfully")
                
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            raise CLIError(f"Error processing image: {e}") from e

    def _get_ocr_target_language(self, args: argparse.Namespace) -> str:
        """Extract target language for OCR processing.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Target language string
            
        Raises:
            CLIError: If language code is missing or invalid
        """
        language_code = getattr(args, 'language_code', None)
        
        if not language_code:
            raise CLIError(
                "Target language code is required for OCR (e.g., 'E' for English, 'C' for Chinese)"
            )
        
        # Accept either single language code or translation pair (use target language)
        if isinstance(language_code, tuple):
            lang_tuple: tuple[str, str] = language_code  # type: ignore[assignment]
            return lang_tuple[1]
        else:
            # Single language string
            return language_code  # type: ignore[return-value]
    
    def _get_translation_languages(self, args: argparse.Namespace) -> tuple[str, str]:
        """Extract source and target languages for translation.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Tuple of (source_language, target_language)
            
        Raises:
            CLIError: If language code is missing or invalid
        """
        language_code = getattr(args, 'language_code', None)
        
        if not language_code:
            raise CLIError("Language code is required for translation")
        
        if not isinstance(language_code, tuple):
            raise CLIError(
                "Translation requires a 2-character language code (e.g., CE, JE, KE)"
            )
        
        # Type narrowing: we know it's a tuple here
        lang_tuple: tuple[str, str] = language_code  # type: ignore[assignment]
        return (lang_tuple[0], lang_tuple[1])

    def _resolve_output_path(self, args: argparse.Namespace) -> Optional[str]:
        """Resolve the output file path based on arguments.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Absolute path to output file or None
        """
        output_file_arg: Optional[str] = getattr(args, 'output_file', None)
        input_file_arg: Optional[str] = getattr(args, 'input_file', None)
        
        if output_file_arg:
            # If output file is a relative path, make it relative to input file directory
            if not os.path.isabs(output_file_arg) and input_file_arg:
                input_dir = os.path.dirname(os.path.abspath(input_file_arg))
                return os.path.join(input_dir, output_file_arg)
            else:
                return os.path.abspath(output_file_arg)
        elif input_file_arg:
            input_dir = os.path.dirname(os.path.abspath(input_file_arg))
            input_name, _ = os.path.splitext(os.path.basename(input_file_arg))
            return os.path.join(input_dir, f"{input_name}_translated.txt")
        
        return None

    def run(self, args: argparse.Namespace) -> None:
        """Run the translation application with the given arguments."""
        try:
            # Validate that some input method is specified
            if not args.input_file and not args.custom_text:
                raise CLIError("Please specify a command (translation, usage report, etc.)")
            
            # Handle input file processing (OCR or translation)
            if args.input_file:
                file_type = self._detect_and_validate_file(args.input_file)
                
                # Route based on file type
                if file_type == 'image':
                    target_language = self._get_ocr_target_language(args)
                    output_file_arg: Optional[str] = getattr(args, 'output_file', None)
                    self.process_image(os.path.abspath(args.input_file), target_language, output_file_arg)
                else:
                    # PDF, DOCX, or TXT - route to translation
                    source_language, target_language = self._get_translation_languages(args)
                    output_file = self._resolve_output_path(args)
                    
                    self.translate_document(
                        args.input_file, source_language, target_language,
                        args.page_nums, args.abstract, output_file, args.auto_save, 
                        args.progressive_save, args.custom_font
                    )
            
            # Handle custom text translation
            elif args.custom_text:
                source_language, target_language = self._get_translation_languages(args)
                output_file = self._resolve_output_path(args)
                
                self.translate_custom_text(
                    source_language, target_language, output_file, args.auto_save, args.custom_font
                )
        
        except CLIError as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            print("\nOperation cancelled.", file=sys.stderr)
            sys.exit(130)  # Standard exit code for SIGINT
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"Unexpected error: {e}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    """Main entry point for the CLI application."""
    try:
        parser = create_argument_parser()
        args = parser.parse_args()

        # Handle info commands first without API initialization
        if handle_info_commands_without_processor(args):
            return

        # Professor is required for translation/OCR commands
        if not args.professor:
            raise CLIError("Professor name is required for translation and OCR commands.")
        
        # Initialize processor (may raise CLIError)
        sandbox = SandboxProcessor(args.professor, model=args.model)
        
        # Run the application (handles all its own exceptions)
        sandbox.run(args)
    
    except CLIError as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
