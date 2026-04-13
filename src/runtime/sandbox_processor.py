"""Runtime processing orchestration for translation and OCR commands."""

import argparse
import logging
import os
import sys
from typing import Optional, List, Tuple, TypedDict, cast

from ..config import get_api_key
from ..errors import CLIError
from ..models import OutputOptions
from ..output.file_output import FileOutputHandler
from ..processors.docx_processor import DocxProcessor
from ..processors.image_processor import ImageProcessor
from ..processors.pdf_processor import PDFProcessor, generate_process_text
from ..processors.txt_processor import TxtProcessor
from ..services.image_processor_service import ImageProcessorService
from ..services.image_translation_service import ImageTranslationService
from ..services.prompt_service import PromptService
from ..services.translation_service import TranslationService
from ..tracking.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


class _SvcKwargs(TypedDict, total=False):
    """Shared keyword arguments passed to every BaseService subclass."""
    token_tracker: TokenTracker
    model: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]


def _parse_page_nums(page_nums_str: Optional[str]) -> Tuple[int, Optional[int]]:
    """Parse a page selection string into zero-based (start, end) indices."""
    if page_nums_str is None:
        return 0, None
    if '-' in page_nums_str:
        start, end = map(int, page_nums_str.split('-'))
        return start - 1, end - 1
    page = int(page_nums_str)
    if page <= 0:
        raise ValueError(f"{page_nums_str} is not a valid page number.")
    return page - 1, page - 1


class SandboxProcessor:
    """Main application class for processing inputs to the Princeton AI Sandbox."""

    def __init__(self, professor_name: str, model: Optional[str] = None,
                 temperature: Optional[float] = None, top_p: Optional[float] = None,
                 max_tokens: Optional[int] = None):
        """Initialize the processor for the specified professor."""
        try:
            api_key, self.professor_display_name = get_api_key(professor_name)
            self.professor_name = professor_name

            logger.debug(f"Initializing processor for professor: {self.professor_display_name}")

            self.token_tracker = TokenTracker(professor=professor_name)
            _svc_kwargs: _SvcKwargs = {"token_tracker": self.token_tracker, "model": model, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
            self.translation_service = TranslationService(api_key, professor_name, **_svc_kwargs)
            self.image_processor_service = ImageProcessorService(api_key, professor_name, **_svc_kwargs)
            self.image_translation_service = ImageTranslationService(api_key, professor_name, **_svc_kwargs)
            self.prompt_service = PromptService(api_key, professor_name, **_svc_kwargs)

            self.image_processor = ImageProcessor()
            self.pdf_processor = PDFProcessor()
            self.file_output = FileOutputHandler()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise CLIError(f"Configuration error: {e}") from e

    def _detect_and_validate_file(self, file_path: str) -> str:
        """Detect file type and validate the file. Caller must pass an absolute path."""
        if not os.path.exists(file_path):
            raise CLIError(f"File '{file_path}' not found.")

        logger.debug(f"Validating file: {file_path}")
        lower_path = file_path.lower()

        if self.image_processor.is_image_file(file_path):
            if not self.image_processor.validate_image_file(file_path):
                raise CLIError(f"Image file '{file_path}' is not valid.")
            logger.debug(f"Detected image file: {file_path}")
            return 'image'
        if lower_path.endswith('.pdf'):
            logger.debug(f"Detected PDF file: {file_path}")
            return 'pdf'
        if lower_path.endswith('.docx'):
            logger.debug(f"Detected Word document: {file_path}")
            return 'docx'
        if lower_path.endswith('.txt'):
            logger.debug(f"Detected text file: {file_path}")
            return 'txt'

        raise CLIError("Unsupported file format. Supported formats: PDF, DOCX, TXT, or image files (JPG, PNG, etc.)")

    def _handle_page_range(self, pages: List[str], page_nums: Optional[str], file_type: str) -> List[str]:
        """Handle page range selection for text-based documents."""
        if not page_nums:
            return pages

        start_page, end_page = _parse_page_nums(page_nums)
        assert end_page is not None  # page_nums is non-empty here, so _parse_page_nums always returns an int

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
        opts: OutputOptions,
    ) -> List[str]:
        """Process text-based files (DOCX, TXT) with common logic."""
        logger.info(f"Processing {file_type.upper()} file: {os.path.basename(file_path)}")

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
            opts,
            file_path,
        )

    def translate_document(
        self,
        file_path: str,
        source_language: str,
        target_language: str,
        page_nums: Optional[str] = None,
        abstract: bool = False,
        opts: OutputOptions = OutputOptions(),
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
                    opts,
                )
            except Exception as e:
                logger.error(f"Error processing image: {e}", exc_info=True)
                raise CLIError(f"Error processing image: {e}") from e
            return

        abstract_text: Optional[str] = self._collect_multiline("Abstract text") or None if abstract else None

        logger.info(f"Starting translation: {source_language} → {target_language}")

        try:
            if file_type == 'pdf':
                with open(file_path, 'rb') as f:
                    pages = self.pdf_processor.process_pdf(f)
                    start_page, end_page = _parse_page_nums(page_nums)
                    document_text = self.translation_service.translate_document(
                        pages,
                        abstract_text,
                        start_page,
                        end_page,
                        source_language,
                        target_language,
                        opts,
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
                    opts,
                )
            else:
                raise CLIError(f"Cannot translate file type '{file_type}'.")

            full_translation = "".join(document_text)


            if not opts.progressive_save and (opts.output_file or opts.auto_save):
                logger.info("Saving translation output")
                self.file_output.save_translation_output(
                    full_translation,
                    file_path,
                    opts.output_file,
                    opts.auto_save,
                    source_language,
                    target_language,
                    opts.custom_font,
                )

        except ImportError as e:
            if "python-docx" in str(e):
                raise CLIError(
                    "python-docx is required to process Word documents. Install it with: pip install python-docx"
                ) from e
            raise CLIError(f"Import error: {e}") from e
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            raise CLIError(f"Error processing document: {e}") from e

    def translate_custom_text(
        self,
        source_language: str,
        target_language: str,
        abstract: bool = False,
        opts: OutputOptions = OutputOptions(),
    ) -> None:
        """Translate custom text input by the user."""
        abstract_text: Optional[str] = self._collect_multiline("Abstract text") or None if abstract else None

        try:
            custom_text = self._collect_multiline(f"Enter the {source_language} text you want to translate to {target_language}")

            if not custom_text.strip():
                logger.warning("No text provided for translation")
                print("No text provided.")
                return

            logger.debug(f"Starting custom text translation: {source_language} -> {target_language}")
            print("\nTranslating...")
            if abstract_text:
                translated_text = self.translation_service.translate_page_text(
                    abstract_text, custom_text, '', source_language, target_language
                )
            else:
                translated_text = self.translation_service.translate_text(custom_text, source_language, target_language)

            if opts.output_file or opts.auto_save:
                input_filename = f"custom_text_{source_language}to{target_language}.txt"
                self.file_output.save_translation_output(
                    translated_text,
                    input_filename,
                    opts.output_file,
                    opts.auto_save,
                    source_language,
                    target_language,
                    opts.custom_font,
                )

        except KeyboardInterrupt:
            logger.info("Translation cancelled by user")
            print("\nTranslation cancelled.")
        except Exception as e:
            logger.error(f"Error during translation: {e}", exc_info=True)
            raise CLIError(f"Error during translation: {e}") from e

    def process_image(self, file_path: str, target_language: str, output_file: Optional[str] = None, vertical: bool = False, passes: int = 1) -> None:
        """Process an image file with OCR (transcribe command)."""
        logger.info(f"Starting OCR processing: {os.path.basename(file_path)} → {target_language}")

        try:
            extracted_text = self.image_processor_service.process_image_ocr(file_path, target_language, output_format="console", vertical=vertical, passes=passes)

            print("\n=== Extracted Text ===")
            print(extracted_text)
            print("======================\n")

            if output_file:
                self.file_output.save_translation_output(
                    extracted_text, file_path, output_file, False,
                    target_language, target_language,
                )

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            raise CLIError(f"Error processing image: {e}") from e

    def process_image_folder(self, folder_path: str, target_language: str, output_file: Optional[str] = None, vertical: bool = False, passes: int = 1) -> None:
        """Process all images in a folder with OCR, printing each result and optionally saving combined output."""
        from ..processors.constants import IMAGE_EXTENSIONS

        folder_path = os.path.abspath(folder_path)
        all_entries = sorted(os.listdir(folder_path))
        image_files = [
            os.path.join(folder_path, name)
            for name in all_entries
            if name.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(folder_path, name))
        ]

        if not image_files:
            raise CLIError(f"No image files found in folder '{folder_path}'.")

        logger.info(f"Processing {len(image_files)} image(s) in folder: {os.path.basename(folder_path)}")
        print(f"Found {len(image_files)} image(s) to process.\n")

        combined_parts: List[str] = []

        for idx, img_path in enumerate(image_files, start=1):
            filename = os.path.basename(img_path)
            print(f"[{idx}/{len(image_files)}] {filename}")
            try:
                extracted_text = self.image_processor_service.process_image_ocr(
                    img_path, target_language, output_format="console", vertical=vertical, passes=passes
                )
            except Exception as e:
                logger.error(f"Error processing '{filename}': {e}", exc_info=True)
                print(f"  ERROR: {e}")
                extracted_text = f"[Error processing {filename}: {e}]"

            print("\n=== Extracted Text ===")
            print(extracted_text)
            print("======================\n")

            combined_parts.append(f"=== {filename} ===\n{extracted_text}")

        if output_file:
            self.file_output.save_translation_output(
                "\n\n".join(combined_parts), None, output_file, False,
                target_language, target_language,
            )

    def process_image_translation(
        self,
        file_path: str,
        source_language: str,
        target_language: str,
        opts: OutputOptions = OutputOptions(),
    ) -> None:
        """Transcribe and translate an image in a single API call (translate command).

        Uses ImageTranslationService to send one combined prompt, allowing
        reasoning models to resolve ambiguous characters using translation context.
        Prints both the transcript and the translation; saves the translation if
        an output path is specified or auto_save is enabled.
        """
        logger.info(
            f"Starting image translation: {os.path.basename(file_path)} "
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

        if opts.output_file or opts.auto_save:
            self.file_output.save_translation_output(
                translation,
                file_path,
                opts.output_file,
                opts.auto_save,
                source_language,
                target_language,
                opts.custom_font,
            )

    def process_prompt(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> None:
        """Send a custom prompt and print (and optionally save) the response."""
        try:
            response = self.prompt_service.send_prompt(user_prompt, system_prompt)
            print("\n" + response)
            if output_file:
                self._save_text_file(response, output_file, "Response")
        except Exception as e:
            logger.error(f"Error sending prompt: {e}", exc_info=True)
            raise CLIError(f"Error sending prompt: {e}") from e

    @staticmethod
    def _save_text_file(text: str, output_file: str, label: str = "Output") -> None:
        """Write *text* to *output_file*, then print the saved path."""
        output_path = os.path.abspath(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"{label} saved to: {os.path.basename(output_path)}")

    @staticmethod
    def _collect_multiline(label: str) -> str:
        """Print a prompt label and collect lines until '---' or EOF."""
        print(f"{label} (type --- on its own line when done):")
        lines: list[str] = []
        while True:
            try:
                line = input()
                if line.strip() == '---':
                    break
                lines.append(line)
            except EOFError:
                break
        return '\n'.join(lines)

    @staticmethod
    def _collect_notes() -> Tuple[Optional[str], Optional[str]]:
        """Ask the user which prompt(s) to annotate, collect their note text, and return
        (system_note, user_note).  Either value is None if not requested."""
        while True:
            try:
                target = input("Add notes to (system / user / both): ").strip().lower()
            except EOFError:
                return None, None
            if target in ('system', 'user', 'both'):
                break
            print("Please enter 'system', 'user', or 'both'.")

        note_text = SandboxProcessor._collect_multiline("Notes")
        if not note_text.strip():
            return None, None

        system_note = note_text if target in ('system', 'both') else None
        user_note   = note_text if target in ('user',   'both') else None
        return system_note, user_note

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

    def _run_translate(self, args: argparse.Namespace) -> None:
        """Handle the 'translate' command."""
        language_code = args.language_code

        if not isinstance(language_code, tuple):
            raise CLIError("Translation requires a 2-character language code (e.g., CE, JE, KE)")

        lang_tuple = cast(Tuple[str, str], language_code)
        if len(lang_tuple) != 2:
            raise CLIError("Translation requires a 2-character language code (e.g., CE, JE, KE)")

        source_language: str = lang_tuple[0]
        target_language: str = lang_tuple[1]

        if getattr(args, 'notes', False):
            sys_note, usr_note = self._collect_notes()
            self.translation_service.system_note = sys_note
            self.translation_service.user_note = usr_note
            self.image_translation_service.system_note = sys_note
            self.image_translation_service.user_note = usr_note

        if getattr(args, 'dry_run', False):
            model_dr = self.translation_service._get_model()
            abstract_text_dr: Optional[str] = None
            if getattr(args, 'abstract', False):
                abstract_text_dr = self._collect_multiline("Abstract text") or None

            if args.input_file:
                file_path_dr = os.path.abspath(args.input_file)
                file_type_dr = self._detect_and_validate_file(file_path_dr)
                if file_type_dr == 'image':
                    sys_p, usr_p = self.image_translation_service.build_prompts(source_language, target_language)
                    self._dry_run_display(
                        self.image_translation_service._get_model(), sys_p, usr_p,
                        note="Image content would be base64-encoded and attached to the user message",
                    )
                    return
                elif file_type_dr == 'pdf':
                    with open(file_path_dr, 'rb') as f:
                        first_page = next(iter(self.pdf_processor.process_pdf(f)), None)
                        page_text_dr = self.pdf_processor.process_page(first_page) if first_page else "[no text found in PDF]"
                elif file_type_dr == 'docx':
                    with open(file_path_dr, 'rb') as f:
                        pages_dr = DocxProcessor.process_docx_with_pages(f, target_page_size=2000)
                        page_text_dr = pages_dr[0] if pages_dr else "[no text found in document]"
                elif file_type_dr == 'txt':
                    with open(file_path_dr, 'r', encoding='utf-8') as f:
                        pages_dr = TxtProcessor.process_txt_with_pages(f, target_page_size=2000)
                        page_text_dr = pages_dr[0] if pages_dr else "[no text found in file]"
                else:
                    page_text_dr = f"[{source_language} text to translate]"
            elif args.custom_text:
                page_text_dr = self._collect_multiline(
                    f"Enter the {source_language} text you want to translate to {target_language}"
                )
                if not page_text_dr.strip():
                    page_text_dr = f"[{source_language} text to translate]"
            else:
                page_text_dr = f"[{source_language} text to translate]"

            combined = generate_process_text(abstract_text_dr or "", page_text_dr, "")
            sys_p, usr_p = self.translation_service.build_prompts(combined, source_language, target_language)
            self._dry_run_display(model_dr, sys_p, usr_p)
            return

        opts = OutputOptions(
            output_file=self._resolve_output_path(args),
            auto_save=getattr(args, 'auto_save', False),
            progressive_save=getattr(args, 'progressive_save', False),
            custom_font=getattr(args, 'custom_font', None),
        )
        if args.custom_text:
            self.translate_custom_text(
                source_language,
                target_language,
                getattr(args, 'abstract', False),
                opts,
            )
        elif args.input_file:
            self.translate_document(
                args.input_file,
                source_language,
                target_language,
                getattr(args, 'page_nums', None),
                getattr(args, 'abstract', False),
                opts,
            )
        else:
            raise CLIError("No input specified. Use -i for file input or -c for custom text.")

    def _run_transcribe(self, args: argparse.Namespace) -> None:
        """Handle the 'transcribe' command."""
        # language_code is already resolved to a full name by parse_single_language_code
        target_language: str = args.language_code

        if getattr(args, 'notes', False):
            sys_note, usr_note = self._collect_notes()
            self.image_processor_service.system_note = sys_note
            self.image_processor_service.user_note = usr_note

        if getattr(args, 'dry_run', False):
            vertical_dr = getattr(args, 'vertical', False)
            model_dr = self.image_processor_service._get_model()
            sys_p, usr_p = self.image_processor_service.build_prompts(target_language, vertical=vertical_dr)
            self._dry_run_display(model_dr, sys_p, usr_p,
                                  note="Image content would be base64-encoded and attached to the user message")
            return

        if not args.input_file:
            raise CLIError("Input file is required for transcribe command. Use -i option.")

        input_path = os.path.abspath(args.input_file)
        output_file = getattr(args, 'output_file', None)
        vertical = getattr(args, 'vertical', False)
        passes = getattr(args, 'passes', 1)
        if passes < 1:
            raise CLIError("--passes must be at least 1.")

        if os.path.isdir(input_path):
            self.process_image_folder(input_path, target_language, output_file, vertical=vertical, passes=passes)
        else:
            file_type = self._detect_and_validate_file(input_path)
            if file_type != 'image':
                raise CLIError(f"Transcribe command requires an image file or folder, but got {file_type}.")
            self.process_image(input_path, target_language, output_file, vertical=vertical, passes=passes)

    def _run_prompt(self, args: argparse.Namespace) -> None:
        """Handle the 'prompt' command."""
        system_prompt_text: Optional[str] = None
        if getattr(args, 'include_system_prompt', False):
            system_prompt_text = self._collect_multiline("System prompt") or None

        if getattr(args, 'dry_run', False):
            model_dr = self.prompt_service._get_model()
            sys_p, usr_p = self.prompt_service.build_prompts(
                "[Interactive prompt — text would be entered at runtime]",
                system_prompt_text,
            )
            self._dry_run_display(model_dr, sys_p, usr_p)
            return

        user_prompt_text = self._collect_multiline("User prompt")
        if not user_prompt_text.strip():
            raise CLIError("No prompt text provided.")

        output_file_p = getattr(args, 'output_file', None)
        self.process_prompt(user_prompt_text, system_prompt_text, output_file_p)

    def run(self, args: argparse.Namespace) -> None:
        """Run the translation application with the given arguments."""
        _dispatch = {
            'translate': self._run_translate,
            'transcribe': self._run_transcribe,
            'prompt': self._run_prompt,
        }
        try:
            handler = _dispatch.get(args.command)
            if handler is None:
                raise CLIError(f"Unknown command: {args.command}")
            handler(args)
        except CLIError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nOperation cancelled.", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            sys.exit(1)

