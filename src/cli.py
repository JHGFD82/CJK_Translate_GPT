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
        description='Translate documents between Chinese, Japanese, Korean, and English using Princeton AI Sandbox',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Language codes:
  Use two characters for translation (source+target): CE, JK, EC, etc.
  Use one character for transcription: E (English), C (Chinese), J (Japanese), K (Korean)

Examples:
  # Global commands (no professor required)
  python main.py --list-models
  python main.py --update-pricing gpt-4o 0.03 0.06
  
  # Usage commands
  python main.py heller usage report
  python main.py heller usage daily
  python main.py heller usage daily 2024-12-25
  
  # Translation commands
  python main.py heller translate CE -i document.pdf
  python main.py heller translate JE -i document.docx -p 1-5
  python main.py heller translate KE -c
  
  # Transcription commands (OCR)
  python main.py heller transcribe E -i image.jpg -o output.txt
        """,
    )

    # Global commands (no professor required)
    parser.add_argument(
        '--list-models',
        dest='list_models',
        action='store_true',
        help='List all available models and their capabilities',
    )
    parser.add_argument(
        '--update-pricing',
        dest='update_pricing',
        type=str,
        nargs=3,
        metavar=('MODEL', 'INPUT_PRICE', 'OUTPUT_PRICE'),
        help='Update pricing for a specific model',
    )

    # Professor-based commands use subparsers
    parser.add_argument(
        'professor',
        type=str,
        nargs='?',
        help='Professor name for API key lookup',
    )

    # Add subparsers for commands (usage, translate, transcribe)
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # ===== USAGE COMMAND =====
    usage_parser = subparsers.add_parser('usage', help='View token usage and costs')
    usage_subparsers = usage_parser.add_subparsers(dest='usage_subcommand', help='Usage subcommand')

    # usage report
    usage_subparsers.add_parser('report', help='Display comprehensive usage report')

    # usage daily [date]
    daily_parser = usage_subparsers.add_parser('daily', help='Display daily usage')
    daily_parser.add_argument(
        'date',
        type=str,
        nargs='?',
        default='today',
        help='Date in YYYY-MM-DD format (defaults to today)',
    )

    # ===== TRANSLATE COMMAND =====
    translate_parser = subparsers.add_parser('translate', help='Translate documents or text')
    translate_parser.add_argument(
        'language_code',
        type=parse_language_code,
        help='Translation direction (CE, JE, KE, etc.)',
    )

    # Input options
    input_group = translate_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', dest='input_file', type=str, help='Input file path (PDF, DOCX, TXT)')
    input_group.add_argument('-c', '--custom', dest='custom_text', action='store_true', help='Input custom text')

    # Translation options
    translate_parser.add_argument(
        '-p', '--page_nums',
        dest='page_nums',
        type=validate_page_nums,
        help='Page numbers to process (e.g., "1" or "1-5")',
    )
    translate_parser.add_argument('-a', '--abstract', dest='abstract', action='store_true', help='Text has an abstract')
    translate_parser.add_argument('-o', '--output', dest='output_file', type=str, help='Output file path (.txt, .pdf, or .docx)')
    translate_parser.add_argument('--auto-save', dest='auto_save', action='store_true', help='Auto-save with timestamp')
    translate_parser.add_argument(
        '--progressive-save',
        dest='progressive_save',
        action='store_true',
        help='Save each page immediately (text output only)',
    )
    translate_parser.add_argument('-f', '--font', dest='custom_font', type=str, help='Custom font name (must be in fonts/)')
    translate_parser.add_argument('-m', '--model', dest='model', type=str, help='Model to use (e.g., gpt-4o, gpt-4o-mini)')

    # ===== TRANSCRIBE COMMAND =====
    transcribe_parser = subparsers.add_parser('transcribe', help='Transcribe images using OCR')
    transcribe_parser.add_argument(
        'language_code',
        type=str,
        help='Target language (E, C, J, or K)',
    )
    transcribe_parser.add_argument('-i', '--input', dest='input_file', type=str, required=True, help='Input image file path')
    transcribe_parser.add_argument('-o', '--output', dest='output_file', type=str, help='Output file path')
    transcribe_parser.add_argument('-m', '--model', dest='model', type=str, help='Model to use (e.g., gpt-4o, gpt-4o-mini)')

    return parser


def main() -> None:
    """Main entry point for the CLI application."""
    setup_logging()

    try:
        parser = create_argument_parser()
        args = parser.parse_args()

        # Handle global commands (no professor required)
        if args.list_models or args.update_pricing:
            if handle_info_commands(args):
                return

        # All other commands require professor name
        if not args.professor:
            raise CLIError(
                "Professor name is required.\n"
                "Usage: python main.py <professor_name> <command> [options]\n"
                "Or for global commands: python main.py --list-models"
            )

        # Handle professor-specific commands
        if not args.command:
            raise CLIError(
                f"No command specified for professor '{args.professor}'.\n"
                "Available commands: usage, translate, transcribe\n"
                "Use --help for more information."
            )

        # Route to appropriate handler
        if args.command == 'usage':
            if handle_info_commands(args):
                return
        elif args.command in ('translate', 'transcribe'):
            model = getattr(args, 'model', None)
            sandbox = SandboxProcessor(args.professor, model=model)
            sandbox.run(args)
        else:
            raise CLIError(f"Unknown command: {args.command}")

    except CLIError as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
