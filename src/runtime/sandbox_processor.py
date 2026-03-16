"""Runtime processing orchestration for translation and OCR commands."""

import argparse
import logging
import os
import sys
from typing import Optional, List, Tuple, cast

from ..config import get_api_key
from ..errors import CLIError
from ..output.file_output import FileOutputHandler
from ..processors.docx_processor import DocxProcessor
from ..processors.image_processor import ImageProcessor
from ..processors.pdf_processor import generate_process_text
from ..processors.txt_processor import TxtProcessor
from ..services.image_processor_service import ImageProcessorService
from ..services.image_translation_service import ImageTranslationService
from ..services.translation_service import TranslationService
from ..tracking.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


def _parse_page_nums(page_nums_str: Optional[str]) -> Tuple[int, int]:
    """Parse a page selection string into zero-based (start, end) indices."""
    if page_nums_str is None:
        return 0, 0
    if '-' in page_nums_str:
        start, end = map(int, page_nums_str.split('-'))
        return start - 1, end - 1
    page = int(page_nums_str)
    if page <= 0:
        raise ValueError(f"{page_nums_str} is not a valid page number.")
    return page - 1, page - 1


class SandboxProcessor:
    """Main application class for processing inputs to the Princeton AI Sandbox."""

    def __init__(self, professor_name: str, model: Optional[str] = None):
        """Initialize the processor for the specified professor."""
        try:
            api_key, self.professor_display_name = get_api_key(professor_name)
            self.professor_name = professor_name

            logger.info(f"Initializing processor for professor: {self.professor_display_name}")

            self.token_tracker = TokenTracker(professor=professor_name)
            self.translation_service = TranslationService(
                api_key, professor_name, token_tracker=self.token_tracker, model=model
            )
            self.image_processor_service = ImageProcessorService(
                api_key, professor_name, token_tracker=self.token_tracker, model=model
            )
            self.image_translation_service = ImageTranslationService(
                api_key, professor_name, token_tracker=self.token_tracker, model=model
            )

            self.image_processor = ImageProcessor()
            self.file_output = FileOutputHandler()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise CLIError(f"Configuration error: {e}") from e

    def _detect_and_validate_file(self, file_path: str) -> str:
        """Detect file type and validate the file."""
        abs_path = os.path.abspath(file_path)

        if not os.path.exists(abs_path):
            raise CLIError(f"File '{file_path}' not found.")

        logger.debug(f"Validating file: {file_path}")
        lower_path = abs_path.lower()

        if self.image_processor.is_image_file(abs_path):
            if not self.image_processor.validate_image_file(abs_path):
                raise CLIError(f"Image file '{file_path}' is not valid.")
            logger.info(f"Detected image file: {file_path}")
            return 'image'
        if lower_path.endswith('.pdf'):
            logger.info(f"Detected PDF file: {file_path}")
            return 'pdf'
        if lower_path.endswith('.docx'):
            logger.info(f"Detected Word document: {file_path}")
            return 'docx'
        if lower_path.endswith('.txt'):
            logger.info(f"Detected text file: {file_path}")
            return 'txt'

        raise CLIError("Unsupported file format. Supported formats: PDF, DOCX, TXT, or image files (JPG, PNG, etc.)")

    def _handle_page_range(self, pages: List[str], page_nums: Optional[str], file_type: str) -> List[str]:
        """Handle page range selection for text-based documents."""
        if not page_nums:
            return pages

        start_page, end_page = _parse_page_nums(page_nums)

        if start_page >= len(pages):
            raise CLIError(f"Page {start_page + 1} does not exist. Document has {len(pages)} logical pages.")

        end_page = min(end_page, len(pages) - 1)
        selected_pages = pages[start_page:end_page + 1]
        logger.info(
            f"Processing pages {start_page + 1}-{end_page + 1} of {file_type} "
            f"(logical pages based on content length)"
        )
        return selected_pages

    def _process_text_based_file(
        self,
        file_path: str,
        file_type: str,
        page_nums: Optional[str],
        abstract_text: Optional[str],
        source_language: str,
        target_language: str,
        output_file: Optional[str],
        auto_save: bool,
        progressive_save: bool,
    ) -> List[str]:
        """Process text-based files (DOCX, TXT) with common logic."""
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
            pages,
            abstract_text,
            source_language,
            target_language,
            output_file,
            auto_save,
            progressive_save,
            file_path,
        )

    def translate_document(
        self,
        file_path: str,
        source_language: str,
        target_language: str,
        page_nums: Optional[str] = None,
        abstract: bool = False,
        output_file: Optional[str] = None,
        auto_save: bool = False,
        progressive_save: bool = False,
        custom_font: Optional[str] = None,
    ) -> None:
        """Translate a document file (PDF, Word document, or text file)."""
        file_path = os.path.abspath(file_path)
        file_type = self._detect_and_validate_file(file_path)

        # Image files bypass the document translation pipeline entirely.
        # A single combined OCR + translation prompt gives reasoning models
        # (e.g. gpt-5) the ability to resolve ambiguous characters using
        # translation context before committing to a transcript.
        if file_type == 'image':
            try:
                self.process_image_translation(
                    file_path,
                    source_language,
                    target_language,
                    output_file,
                    auto_save,
                    custom_font,
                )
            except Exception as e:
                logger.error(f"Error processing image: {e}", exc_info=True)
                raise CLIError(f"Error processing image: {e}") from e
            return

        if abstract:
            print("Enter abstract text, then type --- on its own line to finish:")
            abstract_lines: list[str] = []
            while True:
                try:
                    line = input()
                    if line.strip() == '---':
                        break
                    abstract_lines.append(line)
                except EOFError:
                    break
            abstract_text: Optional[str] = '\n'.join(abstract_lines) or None
        else:
            abstract_text = None

        logger.info(f"Starting translation: {source_language} → {target_language}")

        try:
            if file_type == 'pdf':
                logger.info(f"Processing PDF file: {file_path}")
                with open(file_path, 'rb') as f:
                    pages = self.translation_service.pdf_processor.process_pdf(f)
                    logger.info("Translating PDF pages")
                    start_page, end_page = _parse_page_nums(page_nums)
                    document_text = self.translation_service.translate_document(
                        pages,
                        abstract_text,
                        start_page,
                        end_page,
                        source_language,
                        target_language,
                        output_file,
                        auto_save,
                        progressive_save,
                        file_path,
                    )
            elif file_type in ('docx', 'txt'):
                document_text = self._process_text_based_file(
                    file_path,
                    file_type,
                    page_nums,
                    abstract_text,
                    source_language,
                    target_language,
                    output_file,
                    auto_save,
                    progressive_save,
                )
            else:
                raise CLIError(f"Cannot translate file type '{file_type}'.")

            full_translation = "".join(document_text)
            print(full_translation)

            if not progressive_save and (output_file or auto_save):
                logger.info("Saving translation output")
                self.file_output.save_translation_output(
                    full_translation,
                    file_path,
                    output_file,
                    auto_save,
                    source_language,
                    target_language,
                    custom_font,
                )

            logger.info("Translation completed successfully")

        except ImportError as e:
            if "python-docx" in str(e):
                logger.error("python-docx library not found")
                raise CLIError(
                    "python-docx is required to process Word documents. Install it with: pip install python-docx"
                ) from e
            logger.error(f"Import error: {e}")
            raise CLIError(f"Import error: {e}") from e
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            raise CLIError(f"Error processing document: {e}") from e

    def translate_custom_text(
        self,
        source_language: str,
        target_language: str,
        abstract: bool = False,
        output_file: Optional[str] = None,
        auto_save: bool = False,
        custom_font: Optional[str] = None,
    ) -> None:
        """Translate custom text input by the user."""
        abstract_text: Optional[str] = None
        if abstract:
            print("Enter abstract text, then type --- on its own line to finish:")
            lines: list[str] = []
            while True:
                try:
                    line = input()
                    if line.strip() == '---':
                        break
                    lines.append(line)
                except EOFError:
                    break
            abstract_text = '\n'.join(lines) or None

        print(f"Enter the {source_language} text you want to translate to {target_language}:")
        print("(Type --- on its own line when done)")

        try:
            lines2: list[str] = []
            while True:
                try:
                    line = input()
                    if line.strip() == '---':
                        break
                    lines2.append(line)
                except EOFError:
                    break
            custom_text = '\n'.join(lines2)

            if not custom_text.strip():
                logger.warning("No text provided for translation")
                print("No text provided.")
                return

            logger.info(f"Starting custom text translation: {source_language} -> {target_language}")
            print("\nTranslating...")
            if abstract_text:
                translated_text = self.translation_service.translate_page_text(
                    abstract_text, custom_text, '', source_language, target_language
                )
            else:
                translated_text = self.translation_service.translate_text(custom_text, source_language, target_language)

            if output_file or auto_save:
                input_filename = f"custom_text_{source_language}to{target_language}.txt"
                logger.info("Saving custom text translation")
                self.file_output.save_translation_output(
                    translated_text,
                    input_filename,
                    output_file,
                    auto_save,
                    source_language,
                    target_language,
                    custom_font,
                )

            logger.info("Custom text translation completed successfully")
        except KeyboardInterrupt:
            logger.info("Translation cancelled by user")
            print("\nTranslation cancelled.")
        except Exception as e:
            logger.error(f"Error during translation: {e}", exc_info=True)
            raise CLIError(f"Error during translation: {e}") from e

    def process_image(self, file_path: str, target_language: str, output_file: Optional[str] = None, vertical: bool = False) -> None:
        """Process an image file with OCR (transcribe command)."""
        logger.info(f"Starting OCR processing: {file_path} → {target_language}")

        try:
            extracted_text = self.image_processor_service.process_image_ocr(file_path, target_language, output_format="console", vertical=vertical)

            print("\n=== Extracted Text ===")
            print(extracted_text)
            print("======================\n")

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

    def process_image_translation(
        self,
        file_path: str,
        source_language: str,
        target_language: str,
        output_file: Optional[str] = None,
        auto_save: bool = False,
        custom_font: Optional[str] = None,
    ) -> None:
        """Transcribe and translate an image in a single API call (translate command).

        Uses ImageTranslationService to send one combined prompt, allowing
        reasoning models to resolve ambiguous characters using translation context.
        Prints both the transcript and the translation; saves the translation if
        an output path is specified or auto_save is enabled.
        """
        logger.info(
            f"Starting image translation: {file_path} "
            f"{source_language} → {target_language}"
        )

        transcript, translation = self.image_translation_service.process_image_translation(
            file_path, source_language, target_language
        )

        if transcript:
            print("\n=== Transcript ===")
            print(transcript)
            print("==================\n")

        print("\n=== Translation ===")
        print(translation)
        print("===================\n")

        if output_file or auto_save:
            logger.info("Saving image translation output")
            self.file_output.save_translation_output(
                translation,
                file_path,
                output_file,
                auto_save,
                source_language,
                target_language,
                custom_font,
            )

        logger.info("Image translation completed successfully")

    @staticmethod
    def _dry_run_display(model: str, system_prompt: str, user_prompt: str, note: Optional[str] = None) -> None:
        """Print prompts in a structured format without making any API calls."""
        sep = "=" * 70
        print(f"\n{sep}")
        print("  DRY RUN — No API call will be made")
        print(f"  Model: {model}")
        if note:
            print(f"  Note:  {note}")
        print(sep)
        print("\n--- SYSTEM PROMPT " + "-" * 52)
        print(system_prompt)
        print("\n--- USER PROMPT " + "-" * 54)
        print(user_prompt)
        print(f"\n{sep}\n")

    def _resolve_output_path(self, args: argparse.Namespace) -> Optional[str]:
        """Resolve output file path based on arguments."""
        output_file_arg: Optional[str] = getattr(args, 'output_file', None)
        input_file_arg: Optional[str] = getattr(args, 'input_file', None)

        if output_file_arg:
            if not os.path.isabs(output_file_arg) and input_file_arg:
                input_dir = os.path.dirname(os.path.abspath(input_file_arg))
                return os.path.join(input_dir, output_file_arg)
            return os.path.abspath(output_file_arg)

        return None

    def run(self, args: argparse.Namespace) -> None:
        """Run the translation application with the given arguments."""
        try:
            command = args.command

            if command == 'translate':
                # Handle translation command
                language_code = args.language_code
                
                # Validate language_code is a tuple with 2 elements
                if not isinstance(language_code, tuple):
                    raise CLIError("Translation requires a 2-character language code (e.g., CE, JE, KE)")
                    
                # Cast to the expected type after validation
                lang_tuple = cast(Tuple[str, str], language_code)
                if len(lang_tuple) != 2:
                    raise CLIError("Translation requires a 2-character language code (e.g., CE, JE, KE)")

                source_language: str = lang_tuple[0]
                target_language: str = lang_tuple[1]

                if getattr(args, 'dry_run', False):
                    model_dr = self.translation_service._get_model()
                    page_placeholder = f"[{source_language} text to translate]"
                    abstract_placeholder = f"[{source_language} abstract text]" if getattr(args, 'abstract', False) else ""
                    combined = generate_process_text(abstract_placeholder, page_placeholder, "")
                    sys_p, usr_p = self.translation_service.build_prompts(combined, source_language, target_language)
                    self._dry_run_display(model_dr, sys_p, usr_p)
                    return

                if args.custom_text:
                    output_file = self._resolve_output_path(args)
                    self.translate_custom_text(
                        source_language,
                        target_language,
                        getattr(args, 'abstract', False),
                        output_file,
                        args.auto_save,
                        getattr(args, 'custom_font', None),
                    )
                elif args.input_file:
                    output_file = self._resolve_output_path(args)
                    self.translate_document(
                        args.input_file,
                        source_language,
                        target_language,
                        getattr(args, 'page_nums', None),
                        getattr(args, 'abstract', False),
                        output_file,
                        getattr(args, 'auto_save', False),
                        getattr(args, 'progressive_save', False),
                        getattr(args, 'custom_font', None),
                    )
                else:
                    raise CLIError("No input specified. Use -i for file input or -c for custom text.")

            elif command == 'transcribe':
                # Handle transcribe (OCR) command
                # language_code is already resolved to a full name by parse_single_language_code
                target_language = args.language_code

                if getattr(args, 'dry_run', False):
                    vertical_dr = getattr(args, 'vertical', False)
                    model_dr = self.image_processor_service._get_model()
                    sys_p, usr_p = self.image_processor_service.build_prompts(target_language, vertical=vertical_dr)
                    self._dry_run_display(model_dr, sys_p, usr_p,
                                          note="Image content would be base64-encoded and attached to the user message")
                    return

                if not args.input_file:
                    raise CLIError("Input file is required for transcribe command. Use -i option.")

                # Validate it's an image file
                file_type = self._detect_and_validate_file(args.input_file)
                if file_type != 'image':
                    raise CLIError(f"Transcribe command requires an image file, but got {file_type}.")

                output_file = getattr(args, 'output_file', None)
                vertical = getattr(args, 'vertical', False)
                self.process_image(os.path.abspath(args.input_file), target_language, output_file, vertical=vertical)

            else:
                raise CLIError(f"Unknown command: {command}")

        except CLIError as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            print("\nOperation cancelled.", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"Unexpected error: {e}", file=sys.stderr)
            sys.exit(1)

