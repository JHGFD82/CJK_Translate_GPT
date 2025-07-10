"""
Command-line interface for the CJK Translation script.
"""

import argparse
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from .utils import validate_page_nums, parse_language_code
from .translation_service import TranslationService
from .file_output import FileOutputHandler

# Load environment variables
load_dotenv()


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_api_key() -> str:
    """Get API key from environment variables."""
    try:
        return os.environ['AI_SANDBOX_KEY']
    except KeyError:
        print("Error: AI_SANDBOX_KEY environment variable not found.")
        print("Please set your API key in the environment variables.")
        exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='Extract text from PDF and translate it between different languages using the GPT engine.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Language Code Examples:
  -CE    Chinese to English
  -JK    Japanese to Korean  
  -EJ    English to Japanese
  -KC    Korean to Chinese
  
Available language codes:
  C = Chinese
  J = Japanese
  K = Korean
  E = English

Output Options:
  -o output.txt       Save translation to specified text file
  -o output.pdf       Save translation to specified PDF file
  --auto-save         Auto-save to timestamped file in source directory
'''
    )

    # Language selection - single argument combining source and target
    parser.add_argument('language_code', type=parse_language_code, metavar='LANG_CODE',
                        help='Two-letter language code: first letter is source, second is target (e.g., CE, JK, EJ)')

    # Input source selection
    source_group = parser.add_mutually_exclusive_group(required=True)
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
                        help='Output file path to save the translation (with .txt or .pdf extension)')

    parser.add_argument('--auto-save', dest='auto_save', action='store_true',
                        help='Automatically save output to a timestamped file in the same directory as input PDF')

    return parser


class CJKTranslator:
    """Main application class for CJK translation."""
    
    def __init__(self):
        self.setup_logging()
        self.api_key = get_api_key()
        self.translation_service = TranslationService(self.api_key)
        self.file_output = FileOutputHandler()
        
    def setup_logging(self) -> None:
        """Set up logging configuration."""
        setup_logging()
    
    def translate_pdf(self, file_path: str, source_language: str, target_language: str,
                     page_nums: Optional[str] = None, abstract: bool = False,
                     output_file: Optional[str] = None, auto_save: bool = False) -> None:
        """Translate a PDF file."""
        abstract_text = input('Enter abstract text: ') if abstract else None
        
        try:
            with open(file_path, 'rb') as f:
                pages = self.translation_service.pdf_processor.process_pdf(f)
                document_text = self.translation_service.translate_document(
                    pages, abstract_text, page_nums, source_language, target_language, output_file, auto_save
                )
            
            # Join all translated content
            full_translation = "".join(document_text)
            
            # Display the translation
            print(full_translation)
            
            # Save the translation if requested
            if output_file or auto_save:
                self.file_output.save_translation_output(
                    full_translation, file_path, output_file, auto_save, source_language, target_language
                )
                
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            exit(1)
        except Exception as e:
            print(f"Error processing PDF: {e}")
            exit(1)
    
    def translate_custom_text(self, source_language: str, target_language: str,
                            output_file: Optional[str] = None, auto_save: bool = False) -> None:
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
                translated_text, input_filename, output_file, auto_save, source_language, target_language
            )
    
    def run(self) -> None:
        """Run the main application."""
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Extract source and target languages from the language code
        source_language, target_language = args.language_code
        
        if args.input_PDF:
            self.translate_pdf(
                args.input_PDF, source_language, target_language,
                args.page_nums, args.abstract, args.output_file, args.auto_save
            )
        elif args.custom_text:
            self.translate_custom_text(
                source_language, target_language, args.output_file, args.auto_save
            )


def main() -> None:
    """Main entry point."""
    translator = CJKTranslator()
    translator.run()


if __name__ == '__main__':
    main()
