"""
Command-line interface for the CJK Translation script.
"""

import argparse
import logging
import os
from typing import Optional, List

from dotenv import load_dotenv

from .utils import parse_language_code, get_api_key, validate_page_nums
from .translation_service import TranslationService
from .file_output import FileOutputHandler
from .docx_processor import DocxProcessor
from .txt_processor import TxtProcessor
from .image_processor import ImageProcessor
from .image_processor_service import ImageProcessorService
from .token_tracker import TokenTracker

# Load environment variables
load_dotenv()


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
    parser.add_argument('professor', type=str,
                        help='Professor name for API key lookup')
    
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


class SandboxProcessor:
    """Main application class for processing inputs to the Princeton AI Sandbox."""
    
    def __init__(self, professor_name: str, model: Optional[str] = None):
        """Initialize the processor for the specified professor.
        
        Args:
            professor_name: Name of the professor
            model: Optional model name to use instead of defaults
        """
        try:
            api_key, self.professor_display_name = get_api_key(professor_name)
            self.professor_name = professor_name
            
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
            print(f"Configuration error: {e}")
            exit(1)
    
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
            SystemExit on validation failure
        """
        abs_path = os.path.abspath(file_path)
        
        # Check if file exists
        if not os.path.exists(abs_path):
            print(f"Error: File '{file_path}' not found.")
            exit(1)
        
        # Detect file type - check extensions directly
        lower_path = abs_path.lower()
        
        if self.image_processor.is_image_file(abs_path):
            if not self.image_processor.validate_image_file(abs_path):
                print(f"Error: Image file '{file_path}' is not valid.")
                exit(1)
            return 'image'
        elif lower_path.endswith('.pdf'):
            return 'pdf'
        elif lower_path.endswith('.docx'):
            return 'docx'
        elif lower_path.endswith('.txt'):
            return 'txt'
        else:
            print(f"Error: Unsupported file format. Supported formats: PDF, DOCX, TXT, or image files (JPG, PNG, etc.)")
            exit(1)

    def _handle_page_range(self, pages: List[str], page_nums: Optional[str], file_type: str) -> List[str]:
        """Handle page range selection for text-based documents.
        
        Args:
            pages: List of pages/sections
            page_nums: Page range string (e.g., "1" or "1-3")
            file_type: Type of file being processed (for error messages)
            
        Returns:
            Filtered list of pages based on page range
        """
        if not page_nums:
            return pages
            
        from .utils import extract_page_nums
        start_page, end_page = extract_page_nums(page_nums)
        
        # Ensure page range is valid
        if start_page >= len(pages):
            print(f"Error: Page {start_page + 1} does not exist. Document has {len(pages)} logical pages.")
            exit(1)
        
        # Limit end_page to available pages
        end_page = min(end_page, len(pages) - 1)
        
        # Select the requested page range
        selected_pages = pages[start_page:end_page + 1]
        print(f"Processing pages {start_page + 1}-{end_page + 1} of {file_type} (logical pages based on content length)")
        
        return selected_pages

    def translate_document(self, file_path: str, source_language: str, target_language: str,
                          page_nums: Optional[str] = None, abstract: bool = False,
                          output_file: Optional[str] = None, auto_save: bool = False, progressive_save: bool = False, 
                          custom_font: Optional[str] = None) -> None:
        """Translate a document file (PDF, Word document, or text file)."""
        # Convert file_path to absolute path and detect type
        file_path = os.path.abspath(file_path)
        file_type = self._detect_and_validate_file(file_path)
        
        abstract_text = input('Enter abstract text: ') if abstract else None
        
        try:
            # Process based on file type
            if file_type == 'pdf':
                with open(file_path, 'rb') as f:
                    pages = self.translation_service.pdf_processor.process_pdf(f)
                    document_text = self.translation_service.translate_document(
                        pages, abstract_text, page_nums, source_language, target_language, output_file, auto_save, progressive_save, file_path
                    )
            elif file_type == 'docx':
                with open(file_path, 'rb') as f:
                    pages = DocxProcessor.process_docx_with_pages(f, target_page_size=2000)
                    pages = self._handle_page_range(pages, page_nums, "Word document")
                    document_text = self.translation_service.translate_text_pages(
                        pages, abstract_text, source_language, target_language, output_file, auto_save, progressive_save, file_path
                    )
            elif file_type == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    pages = TxtProcessor.process_txt_with_pages(f, target_page_size=2000)
                    pages = self._handle_page_range(pages, page_nums, "text file")
                    document_text = self.translation_service.translate_text_pages(
                        pages, abstract_text, source_language, target_language, output_file, auto_save, progressive_save, file_path
                    )
            else:
                # This shouldn't happen due to earlier validation, but handle it gracefully
                print(f"Error: Cannot translate file type '{file_type}'.")
                exit(1)
            
            # Join all translated content
            full_translation = "".join(document_text)
            
            # Display the translation
            print(full_translation)
            
            # Save the translation if requested (skip if progressive saving was used)
            if not progressive_save and (output_file or auto_save):
                self.file_output.save_translation_output(
                    full_translation, file_path, output_file, auto_save, source_language, target_language, custom_font
                )
                
        except ImportError as e:
            if "python-docx" in str(e):
                print("Error: python-docx is required to process Word documents.")
                print("Install it with: pip install python-docx")
            else:
                print(f"Import error: {e}")
            exit(1)
        except Exception as e:
            print(f"Error processing document: {e}")
            exit(1)
    
    def translate_custom_text(self, source_language: str, target_language: str,
                            output_file: Optional[str] = None, auto_save: bool = False, custom_font: Optional[str] = None) -> None:
        """Translate custom text input by the user."""
        print(f"Enter the {source_language} text you want to translate to {target_language}:")
        print("(Press Ctrl+D on Unix/Linux/Mac or Ctrl+Z followed by Enter on Windows to finish)")
        
        try:
            custom_text = ""
            while True:
                try:
                    line = input()
                    custom_text += line + "\n"
                except EOFError:
                    break
            
            if not custom_text.strip():
                print("No text provided.")
                return
            
            print("\nTranslating...")
            translated_text = self.translation_service.translate_text(custom_text, source_language, target_language)
            
            # Save the translation if requested
            if output_file or auto_save:
                input_filename = f"custom_text_{source_language}to{target_language}.txt"
                self.file_output.save_translation_output(
                    translated_text, input_filename, output_file, auto_save, source_language, target_language, custom_font
                )
        except KeyboardInterrupt:
            print("\nTranslation cancelled.")
        except Exception as e:
            print(f"Error during translation: {e}")

    def process_image(self, file_path: str, target_language: str, output_file: Optional[str] = None) -> None:
        """Process an image file with OCR."""
        try:
            print(f"Processing image with OCR: {file_path}")
            print(f"Target language: {target_language}")
            
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
                print(f"Extracted text saved to: {output_path}")
                
        except FileNotFoundError:
            print(f"Error: Image file '{file_path}' not found.")
            exit(1)
        except Exception as e:
            print(f"Error processing image: {e}")
            exit(1)

    def show_usage_report(self) -> None:
        """Display token usage report."""
        self.token_tracker.print_usage_report()

    def show_daily_usage(self, date: Optional[str] = None) -> None:
        """Display daily usage report."""
        if date == 'today':
            usage = self.token_tracker.get_daily_usage()
            print(f"\nToday's usage for {self.professor_display_name}:")
        else:
            usage = self.token_tracker.get_daily_usage(date)
            print(f"\\nUsage for {date} for {self.professor_display_name}:")
        
        if not usage.get('models'):
            print("No usage recorded for this date.")
            return
            
        print(f"Total tokens: {usage['total_tokens']:,}")
        print(f"Total cost: ${usage['total_cost']:.4f}")
        print("\\nBy model:")
        for model, model_usage in usage['models'].items():
            print(f"  {model}: {model_usage['total_tokens']:,} tokens, ${model_usage['total_cost']:.4f}")

    def update_pricing(self, model: str, input_price: str, output_price: str) -> None:
        """Update pricing for a specific model."""
        try:
            input_price_float = float(input_price)
            output_price_float = float(output_price)
            self.token_tracker.update_pricing(model, input_price_float, output_price_float)
            print(f"Updated pricing for {model}: Input=${input_price_float}, Output=${output_price_float}")
        except ValueError:
            print("Error: Prices must be valid numbers")
            exit(1)

    def list_models(self) -> None:
        """List all available models and their capabilities."""
        from .config import load_pricing_config, get_pricing_unit, model_supports_vision
        
        config = load_pricing_config()
        models = config["models"]
        pricing_unit = get_pricing_unit()
        
        print("\n=== Available Models ===")
        print(f"Pricing is per {pricing_unit:,} tokens\n")
        
        for model_name, pricing in sorted(models.items()):
            vision = "✓" if model_supports_vision(model_name) else "✗"
            print(f"{model_name}")
            print(f"  Vision Support: {vision}")
            print(f"  Input:  ${pricing['input']:.3f} per {pricing_unit:,} tokens")
            print(f"  Output: ${pricing['output']:.3f} per {pricing_unit:,} tokens")
            print()

    def _handle_info_commands(self, args: argparse.Namespace) -> bool:
        """Handle information/reporting commands.
        
        Args:
            args: Command-line arguments
            
        Returns:
            True if an info command was handled, False otherwise
        """
        if args.list_models:
            self.list_models()
            return True
        
        if args.usage_report:
            self.show_usage_report()
            return True
        
        if args.daily_usage is not None:
            self.show_daily_usage(args.daily_usage)
            return True
        
        if args.update_pricing:
            model, input_price, output_price = args.update_pricing
            self.update_pricing(model, input_price, output_price)
            return True
        
        return False

    def _get_ocr_target_language(self, args: argparse.Namespace) -> str:
        """Extract target language for OCR processing.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Target language string
        """
        language_code = getattr(args, 'language_code', None)
        
        if not language_code:
            print("Error: Target language code is required for OCR (e.g., 'E' for English, 'C' for Chinese)")
            exit(1)
        
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
        """
        language_code = getattr(args, 'language_code', None)
        
        if not language_code:
            print("Error: Language code is required for translation")
            exit(1)
        
        if not isinstance(language_code, tuple):
            print("Error: Translation requires a 2-character language code (e.g., CE, JE, KE)")
            exit(1)
        
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
        # Handle info commands first (usage reports, model lists, etc.)
        if self._handle_info_commands(args):
            return
        
        # Validate that some input method is specified
        if not args.input_file and not args.custom_text:
            print("Error: Please specify a command (translation, usage report, etc.)")
            exit(1)
        
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


def main() -> None:
    """Main entry point for the CLI application."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    sandbox = SandboxProcessor(args.professor, model=args.model)
    sandbox.run(args)


if __name__ == '__main__':
    main()
