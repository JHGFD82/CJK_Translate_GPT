"""CLI controller: argument parsing, validation, and top-level dispatch."""

import argparse
import logging
import sys

from dotenv import load_dotenv

from .config import parse_language_code, validate_page_nums
from .errors import CLIError
from .runtime import SandboxProcessor, handle_info_commands

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


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
        """,
    )

    parser.add_argument(
        'professor',
        type=str,
        nargs='?',
        help='Professor name for API key lookup (required for translation, OCR, and usage reports)',
    )
    parser.add_argument(
        'language_code',
        type=parse_language_code,
        nargs='?',
        help='Translation direction (CE, JE, KE, etc.)',
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '-i', '--input',
        dest='input_file',
        type=str,
        help='Input file path (PDF, Word document, or text file)',
    )
    input_group.add_argument(
        '-c', '--custom',
        dest='custom_text',
        action='store_true',
        help='Input custom text to be translated',
    )

    parser.add_argument(
        '-p', '--page_nums',
        dest='page_nums',
        type=validate_page_nums,
        help='Page numbers to process\nEnter either a single page number or a range in this format: '
             '[starting page number]-[ending page number]\nNo spaces, letters, commas or other symbols '
             'are allowed. For PDFs: actual page numbers. For Word/text files: logical pages based on content length',
    )
    parser.add_argument('-a', '--abstract', dest='abstract', action='store_true', help='The text has an abstract')
    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        type=str,
        help='Output file path to save the translation (with .txt, .pdf, or .docx extension)',
    )
    parser.add_argument(
        '--auto-save',
        dest='auto_save',
        action='store_true',
        help='Automatically save output to a timestamped file in the same directory as input PDF',
    )
    parser.add_argument(
        '--progressive-save',
        dest='progressive_save',
        action='store_true',
        help='Save each page to output file immediately after translation (useful for error recovery)',
    )
    parser.add_argument(
        '-f', '--font',
        dest='custom_font',
        type=str,
        help='Custom font name to use for PDF and Word document generation (font must be in fonts/ directory)',
    )
    parser.add_argument(
        '-m', '--model',
        dest='model',
        type=str,
        help='Specify which model to use (e.g., gpt-4o, gpt-4o-mini, gpt-5)',
    )

    parser.add_argument('--list-models', dest='list_models', action='store_true', help='List all available models and their capabilities')
    parser.add_argument('--usage-report', dest='usage_report', action='store_true', help='Display token usage and cost report')
    parser.add_argument(
        '--daily-usage',
        dest='daily_usage',
        type=str,
        nargs='?',
        const='today',
        help='Display daily usage for a specific date (YYYY-MM-DD format) or today if no date specified',
    )
    parser.add_argument(
        '--update-pricing',
        dest='update_pricing',
        type=str,
        nargs=3,
        metavar=('MODEL', 'INPUT_PRICE', 'OUTPUT_PRICE'),
        help='Update pricing for a specific model (e.g., --update-pricing gpt-4 0.03 0.06)',
    )

    return parser


def main() -> None:
    """Main entry point for the CLI application."""
    setup_logging()

    try:
        parser = create_argument_parser()
        args = parser.parse_args()

        if handle_info_commands(args):
            return

        if not args.professor:
            raise CLIError("Professor name is required for translation and OCR commands.")

        sandbox = SandboxProcessor(args.professor, model=args.model)
        sandbox.run(args)

    except CLIError as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
