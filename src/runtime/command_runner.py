"""CLI command dispatch mixin for SandboxProcessor.

This module holds the ``_CommandMixin`` class, which translates parsed CLI
arguments into calls on the concrete ``SandboxProcessor`` processing methods.
It is split out from ``sandbox_processor.py`` solely for readability; all
``self.*`` references resolve on the ``SandboxProcessor`` subclass via normal
Python MRO.
"""

import argparse
import logging
import os
import sys
from typing import TYPE_CHECKING, Optional, Tuple, cast

from ..errors import CLIError
from ..models import OutputOptions
from ..processors.docx_processor import DocxProcessor
from ..processors.pdf_processor import generate_process_text
from ..processors.txt_processor import TxtProcessor
from ..settings import DEFAULT_PAGE_SIZE

if TYPE_CHECKING:
    from ..processors.image_processor import ImageProcessor
    from ..processors.pdf_processor import PDFProcessor
    from ..services.image_processor_service import ImageProcessorService
    from ..services.image_translation_service import ImageTranslationService
    from ..services.prompt_service import PromptService
    from ..services.translation_service import TranslationService

logger = logging.getLogger(__name__)


class _CommandMixin:
    """Mixin that adds CLI command dispatch to SandboxProcessor.

    All instance-method references to ``self.*`` resolve on the concrete
    ``SandboxProcessor`` subclass via normal Python MRO.
    """

    if TYPE_CHECKING:
        # Attributes and processing methods provided by the SandboxProcessor subclass.
        image_processor: "ImageProcessor"
        image_translation_service: "ImageTranslationService"
        translation_service: "TranslationService"
        image_processor_service: "ImageProcessorService"
        pdf_processor: "PDFProcessor"
        prompt_service: "PromptService"

        def _detect_and_validate_file(self, file_path: str) -> str: ...
        def translate_custom_text(self, source_language: str, target_language: str, abstract: bool, opts: OutputOptions) -> None: ...
        def process_image_translation_folder(self, folder_path: str, source_language: str, target_language: str, opts: OutputOptions, workers: int = 1) -> None: ...
        def translate_document(self, file_path: str, source_language: str, target_language: str, page_nums: Optional[str], abstract: bool, opts: OutputOptions, workers: int = 1) -> None: ...
        def process_image_folder(self, folder_path: str, target_language: str, output_file: Optional[str] = None, vertical: bool = False, passes: int = 1, workers: int = 1) -> None: ...
        def process_image(self, file_path: str, target_language: str, output_file: Optional[str] = None, vertical: bool = False, passes: int = 1) -> None: ...
        def process_prompt(self, user_prompt: str, system_prompt: Optional[str] = None, output_file: Optional[str] = None) -> None: ...

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
            system_note = _CommandMixin._collect_multiline("System note") or None
            user_note   = _CommandMixin._collect_multiline("User note")   or None
            return system_note, user_note

        note_text = _CommandMixin._collect_multiline("Notes")
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
            # Build a prompt preview so the user can see what they are annotating.
            # Use the image-translation prompts for image inputs, text-translation
            # prompts (with a placeholder) for everything else.
            _preview_sys: Optional[str] = None
            _preview_usr: Optional[str] = None
            if args.input_file:
                _fp = os.path.abspath(args.input_file)
                if os.path.exists(_fp) and self.image_processor.is_image_file(_fp):
                    _preview_sys, _preview_usr = self.image_translation_service.build_prompts(
                        source_language, target_language
                    )
                else:
                    _placeholder = generate_process_text("", f"[{source_language} document text]", "")
                    _preview_sys, _preview_usr = self.translation_service.build_prompts(
                        _placeholder, source_language, target_language
                    )
            else:
                _placeholder = generate_process_text("", f"[{source_language} custom text]", "")
                _preview_sys, _preview_usr = self.translation_service.build_prompts(
                    _placeholder, source_language, target_language
                )
            sys_note, usr_note = self._collect_notes(_preview_sys, _preview_usr)
            self.translation_service.system_note = sys_note
            self.translation_service.user_note = usr_note
            self.image_translation_service.system_note = sys_note
            self.image_translation_service.user_note = usr_note

        # Inline note flags (-ns / -nu / -nb) apply directly without interactive flow.
        # -nb sets both; -ns/-nu set individually (and override -nb for their slot).
        _inline_both = getattr(args, 'note_both', None)
        _inline_sys  = getattr(args, 'note_system', None) or _inline_both
        _inline_usr  = getattr(args, 'note_user', None)   or _inline_both
        if _inline_sys is not None:
            self.translation_service.system_note = _inline_sys
            self.image_translation_service.system_note = _inline_sys
        if _inline_usr is not None:
            self.translation_service.user_note = _inline_usr
            self.image_translation_service.user_note = _inline_usr

        if getattr(args, 'kanbun', False):
            self.translation_service.kanbun = True

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
                        pages_dr = DocxProcessor.process_docx_with_pages(f, target_page_size=DEFAULT_PAGE_SIZE)
                        page_text_dr = pages_dr[0] if pages_dr else "[no text found in document]"
                elif file_type_dr == 'txt':
                    with open(file_path_dr, 'r', encoding='utf-8') as f:
                        pages_dr = TxtProcessor.process_txt_with_pages(f, target_page_size=DEFAULT_PAGE_SIZE)
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
        workers = getattr(args, 'workers', 1)
        if args.custom_text:
            self.translate_custom_text(
                source_language,
                target_language,
                getattr(args, 'abstract', False),
                opts,
            )
        elif args.input_file:
            input_path = os.path.abspath(args.input_file)
            if os.path.isdir(input_path):
                self.process_image_translation_folder(
                    input_path,
                    source_language,
                    target_language,
                    opts,
                    workers=workers,
                )
            else:
                self.translate_document(
                    args.input_file,
                    source_language,
                    target_language,
                    getattr(args, 'page_nums', None),
                    getattr(args, 'abstract', False),
                    opts,
                    workers=workers,
                )
        else:
            raise CLIError("No input specified. Use -i for file input or -c for custom text.")

    def _run_transcribe(self, args: argparse.Namespace) -> None:
        """Handle the 'transcribe' command."""
        # language_code is already resolved to a full name by parse_single_language_code
        target_language: str = args.language_code

        if getattr(args, 'notes', False):
            _vertical_flag = getattr(args, 'vertical', False)
            _preview_sys, _preview_usr = self.image_processor_service.build_prompts(
                target_language, vertical=_vertical_flag
            )
            sys_note, usr_note = self._collect_notes(_preview_sys, _preview_usr)
            self.image_processor_service.system_note = sys_note
            self.image_processor_service.user_note = usr_note

        # Inline note flags (-ns / -nu / -nb) apply directly without interactive flow.
        _inline_both = getattr(args, 'note_both', None)
        _inline_sys  = getattr(args, 'note_system', None) or _inline_both
        _inline_usr  = getattr(args, 'note_user', None)   or _inline_both
        if _inline_sys is not None:
            self.image_processor_service.system_note = _inline_sys
        if _inline_usr is not None:
            self.image_processor_service.user_note = _inline_usr

        if getattr(args, 'kanbun', False):
            self.image_processor_service.kanbun = True

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
        workers = getattr(args, 'workers', 1)
        if passes < 1:
            raise CLIError("--passes must be at least 1.")

        if os.path.isdir(input_path):
            self.process_image_folder(input_path, target_language, output_file, vertical=vertical, passes=passes, workers=workers)
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
