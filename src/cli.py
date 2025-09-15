"""
Command-line interface for the CJK Translation script.
"""

import argparse
import logging
import os
import re
from typing import Optional, Tuple, Dict

from dotenv import load_dotenv

from .config import LANGUAGE_MAP
from .translation_service import TranslationService
from .file_output import FileOutputHandler

# Load environment variables
load_dotenv()


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


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        tuple: (api_key, professor_display_name)
    """
    professors = load_professor_config()
    
    # Try to find professor by safe name first
    safe_name = make_safe_filename(professor_name)
    
    if safe_name not in professors:
        # Try exact match in case user provided the safe name directly
        if professor_name.lower() not in professors:
            available_names = [prof['name'] for prof in professors.values()]
            available_safe_names = list(professors.keys())
            
            print(f"Error: Unknown professor '{professor_name}'.")
            print(f"Available professors: {', '.join(available_names)}")
            print(f"You can use either the full name or safe name: {', '.join(available_safe_names)}")
            print("\nAvailable professors:")
            for safe_name_iter, prof_info in professors.items():
                print(f"  '{prof_info['name']}' (or '{safe_name_iter}')")
            exit(1)
        else:
            safe_name = professor_name.lower()
    
    prof_info = professors[safe_name]
    professor_display_name = prof_info['name']
    primary_key_var = prof_info['primary_key']
    backup_key_var = prof_info['backup_key']
    
    # Try primary key first
    try:
        api_key = os.environ[primary_key_var]
        if api_key:
            print(f"Using primary API key for {professor_display_name}")
            return api_key, professor_display_name
    except KeyError:
        pass
    
    # Try backup key
    try:
        api_key = os.environ[backup_key_var]
        if api_key:
            print(f"Using backup API key for {professor_display_name}")
            return api_key, professor_display_name
    except KeyError:
        pass
    
    # If neither key is found
    print(f"Error: API keys for {professor_display_name} not found in environment variables.")
    print(f"Please set either {primary_key_var} or {backup_key_var} in your .env file.")
    exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='Extract text from PDF and translate it between different languages using the GPT engine.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Usage Examples:
  python main.py heller CE -i document.pdf               # Translate PDF using Heller's keys
  python main.py smith JK -i document.pdf               # Translate PDF using Smith's keys  
  python main.py heller --usage-report                  # Show usage report for Heller
  python main.py smith CE -c                            # Translate custom text using Smith's keys

Language Code Examples:
  CE    Chinese to English
  JK    Japanese to Korean  
  EJ    English to Japanese
  KC    Korean to Chinese
  
Available language codes:
  C = Chinese
  J = Japanese
  K = Korean
  E = English

Professor Configuration:
  Professor names and API keys are configured in .env file
  Use professor names (full name or safe filename) to select the appropriate professor
  Run "python show_config.py" to see available professors
  Safe names are created by converting spaces to underscores (e.g., "John Smith" -> "john_smith")
  
Token Tracking:
  Each professor has separate token usage tracking and monthly budgets
  Usage data is stored in data/token_usage_[safe_name].json files

Output Options:
  -o output.txt       Save translation to specified text file
  -o output.pdf       Save translation to specified PDF file
  --auto-save         Auto-save to timestamped file in source directory
  --progressive-save  Save each page immediately after translation (for error recovery)
'''
    )

    # Required professor selection
    parser.add_argument('professor_name', type=str, metavar='PROFESSOR_NAME',
                        help='Professor name to select appropriate API key (e.g., "heller", "smith")')

    # Language selection - single argument combining source and target
    parser.add_argument('language_code', type=parse_language_code, metavar='LANG_CODE', nargs='?',
                        help='Two-letter language code: first letter is source, second is target (e.g., CE, JK, EJ)')

    # Input source selection
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('-i', '--input_PDF', dest='input_PDF', type=str,
                              help='The name of the input PDF file')
    source_group.add_argument('-c', '--custom_text', dest='custom_text', action='store_true',
                              help='Input custom text to be translated')

    # Optional arguments
    parser.add_argument('-p', '--page_nums', dest='page_nums', type=validate_page_nums,
                        help='Page numbers to output\\nEnter either a single page number or a range in this format: '
                             '[starting page number]-[ending page number]\\nNo spaces, letters, commas or other symbols '
                             'are allowed')

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

    # Token usage commands
    parser.add_argument('--usage-report', dest='usage_report', action='store_true',
                        help='Display token usage and cost report')
    
    parser.add_argument('--daily-usage', dest='daily_usage', type=str, nargs='?', const='today',
                        help='Display daily usage for a specific date (YYYY-MM-DD format) or today if no date specified')
    
    parser.add_argument('--update-pricing', dest='update_pricing', type=str, nargs=3, metavar=('MODEL', 'INPUT_PRICE', 'OUTPUT_PRICE'),
                        help='Update pricing for a model (e.g., --update-pricing gpt-4o 2.75 11.00)')

    return parser


class CJKTranslator:
    """Main application class for CJK translation."""
    
    def __init__(self, professor_name: str):
        self.professor_name = professor_name
        self.setup_logging()
        self.api_key, self.professor_display_name = get_api_key(professor_name)
        
        # Use safe name for file operations
        self.safe_name = make_safe_filename(self.professor_display_name)
        
        self.translation_service = TranslationService(self.api_key, self.safe_name)
        self.file_output = FileOutputHandler()
        
    def setup_logging(self) -> None:
        """Set up logging configuration."""
        setup_logging()
    
    def translate_pdf(self, file_path: str, source_language: str, target_language: str,
                     page_nums: Optional[str] = None, abstract: bool = False,
                     output_file: Optional[str] = None, auto_save: bool = False, progressive_save: bool = False, 
                     custom_font: Optional[str] = None) -> None:
        """Translate a PDF file."""
        abstract_text = input('Enter abstract text: ') if abstract else None
        
        try:
            with open(file_path, 'rb') as f:
                pages = self.translation_service.pdf_processor.process_pdf(f)
                document_text = self.translation_service.translate_document(
                    pages, abstract_text, page_nums, source_language, target_language, output_file, auto_save, progressive_save, file_path
                )
            
            # Join all translated content
            full_translation = "".join(document_text)
            
            # Display the translation
            print(full_translation)
            
            # Save the translation if requested (skip if progressive saving was used)
            if not progressive_save and (output_file or auto_save):
                self.file_output.save_translation_output(
                    full_translation, file_path, output_file, auto_save, source_language, target_language, custom_font
                )
                
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            exit(1)
        except Exception as e:
            print(f"Error processing PDF: {e}")
            exit(1)
    
    def translate_custom_text(self, source_language: str, target_language: str,
                            output_file: Optional[str] = None, auto_save: bool = False, custom_font: Optional[str] = None) -> None:
        """Translate custom text input."""
        text_input = input('Enter custom text to be translated: ')
        
        # Determine output format based on whether file output is requested
        if output_file:
            if output_file.lower().endswith('.pdf'):
                output_format = "pdf"
            elif output_file.lower().endswith('.txt'):
                output_format = "txt"
            else:
                output_format = "file"
        elif auto_save:
            output_format = "txt"  # Auto-save defaults to txt format
        else:
            output_format = "console"
        
        translated_text = self.translation_service.generate_text(
            '', text_input, '', 0, source_language, target_language, output_format
        )
        print(translated_text)
        
        # Save custom text translation if requested
        if output_file or auto_save:
            input_filename = "custom_text_translation"
            self.file_output.save_translation_output(
                translated_text, input_filename, output_file, auto_save, source_language, target_language, custom_font
            )
    
    def show_usage_report(self) -> None:
        """Display token usage report."""
        self.translation_service.print_usage_report()
    
    def show_daily_usage(self, date: Optional[str] = None) -> None:
        """Display daily usage for a specific date or today."""
        if date and date != 'today':
            # Validate date format
            try:
                from datetime import datetime
                datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                print("Error: Date must be in YYYY-MM-DD format")
                return
        
        target_date = None if date == 'today' else date
        usage = self.translation_service.get_daily_usage(target_date)
        
        display_date = date if date != 'today' else 'today'
        print(f"\nDaily Usage Report for {self.professor_display_name} ({display_date}):")
        print("-" * 50)
        print(f"Total Tokens: {usage['total_tokens']:,}")
        print(f"  • Input Tokens: {usage['total_input_tokens']:,}")
        print(f"  • Output Tokens: {usage['total_output_tokens']:,}")
        print(f"Total Cost: ${usage['total_cost']:.4f}")
        print("-" * 50)
    
    def update_pricing(self, model: str, input_price: str, output_price: str) -> None:
        """Update pricing for a model."""
        try:
            input_price_float = float(input_price)
            output_price_float = float(output_price)
            
            self.translation_service.update_model_pricing(model, input_price_float, output_price_float)
            print(f"Updated pricing for {model}:")
            print(f"  Input: ${input_price_float:.3f} per 1M tokens")
            print(f"  Output: ${output_price_float:.3f} per 1M tokens")
            
        except ValueError:
            print("Error: Prices must be valid numbers")
            return
    
    def run(self, args: argparse.Namespace) -> None:
        """Run the main application with parsed arguments."""
        # Handle token usage commands first (these don't require language codes but still need professor)
        if args.usage_report:
            self.show_usage_report()
            return
        
        if args.daily_usage is not None:
            self.show_daily_usage(args.daily_usage)
            return
        
        if args.update_pricing:
            model, input_price, output_price = args.update_pricing
            self.update_pricing(model, input_price, output_price)
            return
        
        # Check if language_code is provided for translation commands
        if not args.language_code:
            if args.input_PDF or args.custom_text:
                print("Error: Language code is required for translation commands")
                exit(1)
            else:
                print("Error: Please specify a command (translation, usage report, etc.)")
                exit(1)
        
        # Extract source and target languages from the language code
        source_language, target_language = args.language_code
        
        if args.input_PDF:
            self.translate_pdf(
                args.input_PDF, source_language, target_language,
                args.page_nums, args.abstract, args.output_file, args.auto_save, args.progressive_save, args.custom_font
            )
        elif args.custom_text:
            self.translate_custom_text(
                source_language, target_language, args.output_file, args.auto_save, args.custom_font
            )


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create translator with professor name parameter
    translator = CJKTranslator(args.professor_name)
    translator.run(args)


if __name__ == '__main__':
    main()
