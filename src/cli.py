"""CLI controller: argument parsing, validation, and top-level dispatch."""

import argparse
import logging
import sys

from dotenv import load_dotenv

from .config import parse_language_code, parse_single_language_code, validate_page_nums
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
        epilog="""\nLanguage codes:
  Two characters for translation (source→target): CE, EC, JE, EJ, KE, EK, JK, KJ, SE, TE, ...
  One character  for transcription (OCR):          E (English), C (Chinese), S (Simplified Chinese),
                                                    T (Traditional Chinese), J (Japanese), K (Korean)

Usage / reporting:
  python main.py heller usage report              Current month report + budget status
  python main.py heller usage report --all-time   Above + all-time totals across all archived months
  python main.py heller usage report 2025-07      Report for a specific archived month
  python main.py heller usage months              List all archived month files
  python main.py heller usage daily               Today's usage
  python main.py heller usage daily 2026-03-01    Usage for a specific date

Global commands (no professor required):
  python main.py --show-config
  python main.py --list-models

Using a new model for the first time:
  python main.py heller translate CE -m openai/gpt-4o-new   Auto-registers from llmprices.ai
  python main.py heller translate CE -m google/gemini-2.5-pro
  Use 'provider/model' format for OpenAI or Google models not yet in model_catalog.json.
  For all other models, add them directly to src/model_catalog.json.

Translation:
  python main.py heller translate CE -i document.pdf
  python main.py heller translate JE -i document.docx -p 1-5
  python main.py heller translate KE -c
  python main.py heller translate CE -i doc.pdf -o translated.docx    Save as Word document
  python main.py heller translate CE -i doc.pdf --progressive-save    Save each page immediately

Transcription (OCR):
  python main.py heller transcribe E -i image.jpg
  python main.py heller transcribe E -i image.jpg -o output.txt
  python main.py heller transcribe S -i scan.png                     Simplified Chinese
  python main.py heller transcribe T -i scan.png -m gpt-4o-mini      Traditional Chinese
  python main.py heller transcribe J -i scan.png                     Japanese (kanji + kana)
  python main.py heller transcribe C -i scan.png -m gpt-4o-mini      Generic Chinese
        """,
    )

    # Global commands (no professor required)
    parser.add_argument(
        '--show-config',
        dest='show_config',
        action='store_true',
        help='Show professor configuration and data-file status',
    )
    parser.add_argument(
        '--list-models',
        dest='list_models',
        action='store_true',
        help='List all available models and their capabilities',
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

    # usage report [YYYY-MM]
    report_parser = usage_subparsers.add_parser(
        'report',
        help='Display usage report (current month by default)',
    )
    report_parser.add_argument(
        'month',
        type=str,
        nargs='?',
        default=None,
        metavar='YYYY-MM',
        help='Archived month to report on (e.g. 2025-07). Omit for current month.',
    )
    report_parser.add_argument(
        '--all-time',
        action='store_true',
        default=False,
        help='Include all-time totals aggregated from all archived months (current month only)',
    )

    # usage months
    usage_subparsers.add_parser('months', help='List all archived months for this professor')

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
    input_group = translate_parser.add_mutually_exclusive_group(required=False)
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
    translate_parser.add_argument(
        '--dry-run',
        dest='dry_run',
        action='store_true',
        help='Print the prompt(s) that would be sent to the model without making any API calls',
    )
    translate_parser.add_argument(
        '--notes',
        dest='notes',
        action='store_true',
        help='Interactively append ad-hoc notes to the system prompt, user prompt, or both before sending',
    )

    # ===== TRANSCRIBE COMMAND =====
    transcribe_parser = subparsers.add_parser('transcribe', help='Transcribe images using OCR')
    transcribe_parser.add_argument(
        'language_code',
        type=parse_single_language_code,
        help='Target language: E (English), C (Chinese), S (Simplified Chinese), T (Traditional Chinese), J (Japanese), K (Korean)',
    )
    transcribe_parser.add_argument('-i', '--input', dest='input_file', type=str, required=False, help='Input image file path')
    transcribe_parser.add_argument('-o', '--output', dest='output_file', type=str, help='Output file path')
    transcribe_parser.add_argument('-m', '--model', dest='model', type=str, help='Model to use (e.g., gpt-4o, gpt-4o-mini)')
    transcribe_parser.add_argument('-v', '--vertical', dest='vertical', action='store_true', help='Text is predominantly vertical (top-to-bottom, right-to-left columns)')
    transcribe_parser.add_argument(
        '--dry-run',
        dest='dry_run',
        action='store_true',
        help='Print the prompt(s) that would be sent to the model without making any API calls',
    )
    transcribe_parser.add_argument(
        '--notes',
        dest='notes',
        action='store_true',
        help='Interactively append ad-hoc notes to the system prompt, user prompt, or both before sending',
    )

    # ===== PROMPT COMMAND =====
    prompt_parser = subparsers.add_parser('prompt', help='Send a custom prompt to the AI model')
    prompt_parser.add_argument(
        '-s', '--system',
        dest='include_system_prompt',
        action='store_true',
        help='Prompt for a system (developer) prompt before the user prompt',
    )
    prompt_parser.add_argument('-o', '--output', dest='output_file', type=str, help='Save response to file')
    prompt_parser.add_argument('-m', '--model', dest='model', type=str, help='Model to use (e.g., gpt-4o, gpt-4o-mini)')
    prompt_parser.add_argument(
        '--dry-run',
        dest='dry_run',
        action='store_true',
        help='Print the prompt(s) that would be sent to the model without making any API calls',
    )

    return parser


def main() -> None:
    """Main entry point for the CLI application."""
    setup_logging()

    try:
        parser = create_argument_parser()
        args = parser.parse_args()

        # Handle global commands (no professor required)
        if args.show_config or args.list_models:
            if handle_info_commands(args):
                return

        # All other commands require professor name
        if not args.professor:
            raise CLIError(
                "Professor name is required.\n"
                "Usage: python main.py <professor_name> <command> [options]\n"
                "\nAvailable commands:\n"
                "  usage report [YYYY-MM] [--all-time]  Token usage report\n"
                "  usage months                         List archived month files\n"
                "  usage daily [YYYY-MM-DD]             Daily usage\n"
                "  translate <code> -i <file>           Translate a document\n"
                "  transcribe <lang> -i <image>         OCR an image\n"
                "\nOr for global commands: python main.py --show-config | --list-models"
            )

        # Handle professor-specific commands
        if not args.command:
            raise CLIError(
                f"No command specified for professor '{args.professor}'.\n"
                "\nAvailable commands:\n"
                "  usage report [YYYY-MM] [--all-time]  Token usage report\n"
                "  usage months                         List archived month files\n"
                "  usage daily [YYYY-MM-DD]             Daily usage\n"
                "  translate <code> -i <file>           Translate a document\n"
                "  transcribe <lang> -i <image>         OCR an image\n"
                "\nRun 'python main.py --help' for full usage information."
            )

        # Route to appropriate handler
        if args.command == 'usage':
            if handle_info_commands(args):
                return
        elif args.command in ('translate', 'transcribe', 'prompt'):
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
