"""Runtime processing orchestration for translation and OCR commands."""

import logging
import os
from typing import Optional, List, Tuple, TypedDict

from tqdm import tqdm

from ..config import get_api_key
from ..errors import CLIError
from ..models import OutputOptions
from ..output.file_output import FileOutputHandler
from ..processors.docx_processor import DocxProcessor
from ..processors.image_processor import ImageProcessor
from ..processors.pdf_processor import PDFProcessor
from ..processors.txt_processor import TxtProcessor
from ..services.image_processor_service import ImageProcessorService
from ..services.image_translation_service import ImageTranslationService
from ..services.parallel_utils import tqdm_logging
from ..services.prompt_service import PromptService
from ..services.translation_service import TranslationService
from ..tracking.token_tracker import TokenTracker
from .command_runner import _CommandMixin

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


class SandboxProcessor(_CommandMixin):
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
        workers: int = 1,
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
            workers=workers,
        )

    def translate_document(
        self,
        file_path: str,
        source_language: str,
        target_language: str,
        page_nums: Optional[str] = None,
        abstract: bool = False,
        opts: OutputOptions = OutputOptions(),
        workers: int = 1,
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
                        workers=workers,
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
                    workers=workers,
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

    def process_image_folder(self, folder_path: str, target_language: str, output_file: Optional[str] = None, vertical: bool = False, passes: int = 1, workers: int = 1) -> None:
        """Process all images in a folder with OCR, printing each result and optionally saving combined output.

        When ``workers > 1`` images are dispatched in parallel via a ThreadPoolExecutor.
        Multi-pass OCR within each image always runs sequentially inside the worker.
        Results are printed and assembled in the original sorted-filename order.
        """
        from ..processors.constants import IMAGE_EXTENSIONS
        from concurrent.futures import ThreadPoolExecutor, as_completed as futures_as_completed

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

        # --- sequential path ---
        if workers <= 1:
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
            return

        # --- parallel path ---
        actual_workers = min(workers, len(image_files))
        if actual_workers < workers:
            logger.info(f"OCR workers capped at {actual_workers} (folder has {len(image_files)} image(s))")

        results_map: dict[int, tuple[str, str]] = {}  # index → (filename, extracted_text)

        # Warm the pricing cache on the main thread so workers share the fast path.
        # Also suppress per-image/per-pass prints that would interleave with tqdm.
        self.image_processor_service._get_model()
        self.image_processor_service._suppress_inline_print = True

        def _ocr_one(idx: int, img_path: str) -> tuple[int, str, str]:
            filename = os.path.basename(img_path)
            extracted = self.image_processor_service.process_image_ocr(
                img_path, target_language, output_format="console", vertical=vertical, passes=passes
            )
            return idx, filename, extracted

        baseline_tokens = self.token_tracker.usage_data["total_usage"].get("total_tokens", 0)
        baseline_cost = self.token_tracker.usage_data["total_usage"].get("total_cost", 0.0)

        with tqdm_logging():
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                future_map = {
                    executor.submit(_ocr_one, i, path): i
                    for i, path in enumerate(image_files)
                }
                desc = f"Transcribing ({actual_workers} workers)... "
                with tqdm(total=len(image_files), desc=desc, ascii=True) as pbar:
                    for future in futures_as_completed(future_map):
                        orig_idx = future_map[future]
                        try:
                            idx, filename, extracted_text = future.result()
                            results_map[idx] = (filename, extracted_text)
                        except Exception as e:
                            filename = os.path.basename(image_files[orig_idx])
                            logger.error(f"Error processing '{filename}': {e}", exc_info=True)
                            results_map[orig_idx] = (filename, f"[Error processing {filename}: {e}]")
                        try:
                            run_tokens = int(self.token_tracker.usage_data["total_usage"].get("total_tokens", 0)) - int(baseline_tokens)
                            run_cost = float(self.token_tracker.usage_data["total_usage"].get("total_cost", 0.0)) - float(baseline_cost)
                            pbar.set_postfix(tokens=f"{run_tokens:,}", cost=f"${run_cost:.4f}")
                        except (TypeError, ValueError):
                            pass
                        pbar.update(1)

        # Print and assemble in sorted-filename (original) order
        combined_parts_p: List[str] = []
        for idx in range(len(image_files)):
            filename, extracted_text = results_map[idx]
            print(f"[{idx + 1}/{len(image_files)}] {filename}")
            print("\n=== Extracted Text ===")
            print(extracted_text)
            print("======================\n")
            combined_parts_p.append(f"=== {filename} ===\n{extracted_text}")

        if output_file:
            self.file_output.save_translation_output(
                "\n\n".join(combined_parts_p), None, output_file, False,
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

    def process_image_translation_folder(
        self,
        folder_path: str,
        source_language: str,
        target_language: str,
        opts: OutputOptions = OutputOptions(),
        workers: int = 1,
    ) -> None:
        """Translate all images in a folder using the combined OCR+translation service.

        When ``workers > 1`` images are dispatched in parallel via a ThreadPoolExecutor.
        Results are printed and assembled in sorted-filename order after all workers finish.
        """
        from ..processors.constants import IMAGE_EXTENSIONS
        from concurrent.futures import ThreadPoolExecutor, as_completed as futures_as_completed

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
        print(f"Found {len(image_files)} image(s) to translate.\n")

        # --- sequential path ---
        if workers <= 1:
            combined_parts: List[str] = []
            for idx, img_path in enumerate(image_files, start=1):
                filename = os.path.basename(img_path)
                print(f"[{idx}/{len(image_files)}] {filename}")
                try:
                    transcript, translation = self.image_translation_service.process_image_translation(
                        img_path, source_language, target_language
                    )
                except Exception as e:
                    logger.error(f"Error processing '{filename}': {e}", exc_info=True)
                    print(f"  ERROR: {e}")
                    transcript, translation = "", f"[Error processing {filename}: {e}]"
                if transcript:
                    print("\n=== Transcript ===")
                    print(transcript)
                    print("==================\n")
                print("\n=== Translation ===")
                print(translation)
                print("===================\n")
                combined_parts.append(f"=== {filename} ===\n{translation}")
            if opts.output_file or opts.auto_save:
                self.file_output.save_translation_output(
                    "\n\n".join(combined_parts), None, opts.output_file, opts.auto_save,
                    source_language, target_language, opts.custom_font,
                )
            return

        # --- parallel path ---
        actual_workers = min(workers, len(image_files))
        if actual_workers < workers:
            logger.info(f"Image translation workers capped at {actual_workers} (folder has {len(image_files)} image(s))")

        results_map: dict[int, tuple[str, str, str]] = {}  # index → (filename, transcript, translation)

        # Warm pricing cache and suppress per-image prints before dispatching workers
        self.image_translation_service._get_model()
        self.image_translation_service._suppress_inline_print = True

        def _translate_one(idx: int, img_path: str) -> tuple[int, str, str, str]:
            filename = os.path.basename(img_path)
            transcript, translation = self.image_translation_service.process_image_translation(
                img_path, source_language, target_language
            )
            return idx, filename, transcript, translation

        baseline_tokens = self.token_tracker.usage_data["total_usage"].get("total_tokens", 0)
        baseline_cost = self.token_tracker.usage_data["total_usage"].get("total_cost", 0.0)

        with tqdm_logging():
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                future_map = {
                    executor.submit(_translate_one, i, path): i
                    for i, path in enumerate(image_files)
                }
                desc = f"Translating ({actual_workers} workers)... "
                with tqdm(total=len(image_files), desc=desc, ascii=True) as pbar:
                    for future in futures_as_completed(future_map):
                        orig_idx = future_map[future]
                        try:
                            idx, filename, transcript, translation = future.result()
                            results_map[idx] = (filename, transcript, translation)
                        except Exception as e:
                            filename = os.path.basename(image_files[orig_idx])
                            logger.error(f"Error processing '{filename}': {e}", exc_info=True)
                            results_map[orig_idx] = (filename, "", f"[Error processing {filename}: {e}]")
                        try:
                            run_tokens = int(self.token_tracker.usage_data["total_usage"].get("total_tokens", 0)) - int(baseline_tokens)
                            run_cost = float(self.token_tracker.usage_data["total_usage"].get("total_cost", 0.0)) - float(baseline_cost)
                            pbar.set_postfix(tokens=f"{run_tokens:,}", cost=f"${run_cost:.4f}")
                        except (TypeError, ValueError):
                            pass
                        pbar.update(1)

        # Print and assemble in sorted-filename (original) order
        combined_parts_p: List[str] = []
        for idx in range(len(image_files)):
            filename, transcript, translation = results_map[idx]
            print(f"[{idx + 1}/{len(image_files)}] {filename}")
            if transcript:
                print("\n=== Transcript ===")
                print(transcript)
                print("==================\n")
            print("\n=== Translation ===")
            print(translation)
            print("===================\n")
            combined_parts_p.append(f"=== {filename} ===\n{translation}")

        if opts.output_file or opts.auto_save:
            self.file_output.save_translation_output(
                "\n\n".join(combined_parts_p), None, opts.output_file, opts.auto_save,
                source_language, target_language, opts.custom_font,
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
    def _collect_notes(
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Optionally display the current prompts, then collect note text to append.

        If *system_prompt* or *user_prompt* are provided they are shown before
        the question so the user has context for what they are annotating.

        Options:
          system   — one note appended to the system prompt only
          user     — one note appended to the user prompt only
          both     — the same note appended to both prompts
          separate — different notes collected individually for system then user
        """
        if system_prompt is not None or user_prompt is not None:
            sep = "-" * 70
            print(f"\n{sep}")
            print("  CURRENT PROMPTS  (your notes will be appended to these)")
            print(sep)
            if system_prompt is not None:
                print("\n--- SYSTEM PROMPT ---")
                print(system_prompt)
            if user_prompt is not None:
                print("\n--- USER PROMPT ---")
                print(user_prompt)
            print(f"\n{sep}\n")

        while True:
            try:
                target = input("Add notes to (system / user / both / separate): ").strip().lower()
            except EOFError:
                return None, None
            if target in ('system', 'user', 'both', 'separate'):
                break
            print("Please enter 'system', 'user', 'both', or 'separate'.")

        if target == 'separate':
            system_note = SandboxProcessor._collect_multiline("System note") or None
            user_note   = SandboxProcessor._collect_multiline("User note")   or None
            return system_note, user_note

        note_text = SandboxProcessor._collect_multiline("Notes")
        if not note_text.strip():
            return None, None

        system_note = note_text if target in ('system', 'both') else None
        user_note   = note_text if target in ('user',   'both') else None
        return system_note, user_note

