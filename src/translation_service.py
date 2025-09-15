"""
Translation service for the CJK Translation script.
"""

import logging
import time
from typing import List, Optional, Iterable, Dict, Any
from itertools import islice
from tqdm import tqdm

from openai import AzureOpenAI
from pdfminer.pdfpage import PDFPage

from .config import (
    get_available_models, DEFAULT_MODEL, SANDBOX_API_VERSION, SANDBOX_ENDPOINT,
    TRANSLATION_TEMPERATURE, TRANSLATION_MAX_TOKENS, TRANSLATION_TOP_P, CONTEXT_PERCENTAGE,
    PAGE_DELAY_SECONDS, MAX_RETRIES, BASE_RETRY_DELAY
)
from .pdf_processor import PDFProcessor, extract_page_nums, generate_process_text
from .token_tracker import TokenTracker


class TranslationService:
    """Handles translation operations using OpenAI API."""
    
    def __init__(self, api_key: str, professor: Optional[str] = None, token_tracker_file: Optional[str] = None):
        self.api_key = api_key
        self.professor = professor
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=SANDBOX_ENDPOINT,
            api_version=SANDBOX_API_VERSION
        )
        self.pdf_processor = PDFProcessor()
        self.token_tracker = TokenTracker(professor=professor, data_file=token_tracker_file)
    
    def _get_model(self) -> str:
        """Get the default model, with fallback if not available."""
        available_models = get_available_models()
        return DEFAULT_MODEL if DEFAULT_MODEL in available_models else available_models[0]
    
    def _create_translation_prompt(self, source_language: str, target_language: str, output_format: str = "console") -> tuple[str, str]:
        """Create system and user prompt templates for translation."""
        
        # Determine formatting instructions based on output format
        if output_format.lower() in ["pdf", "txt", "file"]:
            formatting_instruction = (
                f'Use proper paragraph breaks and standard text formatting suitable for file output. '
                f'Use actual line breaks (not \\n characters) to separate paragraphs and sections naturally.'
            )
        else:  # console output
            formatting_instruction = (
                f'You can format and line break the output yourself using "\\n" for line breaks in console output.'
            )
        
        system_prompt = (
            f'Follow the instructions carefully. Please act as a professional translator from {source_language} '
            f'to {target_language}. I will provide you with text from a PDF document, and your task is '
            f'to translate it from {source_language} to {target_language}. Please only output the translation and do not '
            f'output any irrelevant content. If there are garbled characters or other non-standard text '
            f'content, delete the garbled characters. '
            f'{formatting_instruction} '
            f'You may be provided with "--Context: " and the text from either the document\'s abstract or '
            f'a sample of text from the previous page. You will also be provided with "--Current Page: " '
            f'which includes the OCR characters of the current page. Only output the {target_language} translation of '
            f'the "--Current Page: ". Do not output the context, nor the "--Context: " and "--Current Page: " '
            f'labels.'
        )
        
        user_prompt_template = (
            f'Translate only the {source_language} text of the "--Current Page: " to {target_language}, without outputting any other '
            f'content, and without outputting anything related to "--Context: ", if provided. Do not provide '
            f'any prompts to the user, for example: "This is the translation of the current page.":\n'
        )
        
        return system_prompt, user_prompt_template
    
    def translate_text(self, text: str, source_language: str, target_language: str, output_format: str = "console") -> str:
        """Translate text using the specified model with retry logic for content filters."""
        model = self._get_model()
        system_prompt, user_prompt_template = self._create_translation_prompt(source_language, target_language, output_format)
        user_prompt = user_prompt_template + text
        
        # Retry logic for content filter issues
        max_retries = MAX_RETRIES
        base_delay = BASE_RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + (0.1 * attempt)
                    logging.info(f'Retrying API call (attempt {attempt + 1}/{max_retries}) after {delay:.1f}s delay...')
                    time.sleep(delay)
                
                logging.info(f'Making API call to model: {model}')
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=TRANSLATION_TEMPERATURE,
                    max_tokens=TRANSLATION_MAX_TOKENS,
                    top_p=TRANSLATION_TOP_P,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                
                # Log response details
                logging.info(f'API call successful. Response ID: {response.id}')
                logging.info(f'Model used: {response.model}')
                
                # Log token usage if available
                if response.usage:
                    # Record token usage
                    usage = self.token_tracker.record_usage(
                        model=response.model,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        requested_model=model
                    )
                    
                    logging.info(f'Tokens used - Prompt: {response.usage.prompt_tokens}, '
                               f'Completion: {response.usage.completion_tokens}, '
                               f'Total: {response.usage.total_tokens}, '
                               f'Cost: ${usage.total_cost:.4f}')
                else:
                    logging.warning('No token usage information available in response.')
                
                content = response.choices[0].message.content
                if content is not None:
                    print("\n" + content)
                    logging.info('Translation completed successfully.')
                    return content
                else:
                    print("\n[No content returned by the model]")
                    logging.warning('No content returned by the model.')
                    return ""
                    
            except Exception as e:
                error_result = self._handle_translation_error(e)
                
                # If it's a content filter issue and we have retries left, try again
                if error_result == "content_filter_triggered" and attempt < max_retries - 1:
                    logging.warning(f'Content filter triggered on attempt {attempt + 1}, retrying...')
                    continue
                elif error_result == "content_filter_triggered":
                    logging.error(f'Content filter triggered after {max_retries} attempts, skipping this text')
                    return ""
                else:
                    # For other errors, return the result or re-raise
                    return error_result
        
        # This should never be reached, but just in case
        return ""
    
    def _handle_translation_error(self, error: Exception) -> str:
        """Handle translation errors and return appropriate response."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Check for specific OpenAI error types
        if "context_length_exceeded" in error_message.lower() or "maximum context length" in error_message.lower():
            logging.error(f'Context length exceeded: {error_message}')
            return "context_length_exceeded"
        elif "rate_limit" in error_message.lower():
            logging.error(f'Rate limit exceeded: {error_message}')
            raise Exception(f'Rate limit exceeded: {error_message}')
        elif "invalid_request" in error_message.lower():
            logging.error(f'Invalid request: {error_message}')
            raise Exception(f'Invalid request: {error_message}')
        elif "authentication" in error_message.lower() or "unauthorized" in error_message.lower():
            logging.error(f'Authentication error: {error_message}')
            raise Exception(f'Authentication error: {error_message}')
        elif "content_filter" in error_message.lower() or "jailbreak" in error_message.lower():
            logging.error(f'Content filter triggered: {error_message}')
            return "content_filter_triggered"
        else:
            logging.error(f'API call failed with {error_type}: {error_message}')
            raise Exception(f'API call failed with {error_type}: {error_message}')
    
    def translate_page_text(self, abstract_text: str, page_text: str, previous_page: str, 
                          source_language: str, target_language: str, output_format: str = "console") -> str:
        """Translate page text with context."""
        process_text = generate_process_text(abstract_text, page_text, previous_page, CONTEXT_PERCENTAGE)
        return self.translate_text(process_text, source_language, target_language, output_format)
    
    def generate_text(self, abstract_text: str, page_text: str, previous_page: str, 
                     page_num: int, source_language: str, target_language: str, output_format: str = "console") -> str:
        """Generate translated text for a page, handling context length limits."""
        result: list[str] = []
        parts_to_translate = [page_text]

        while parts_to_translate:
            current_part = parts_to_translate.pop()
            translated_text = self.translate_page_text(
                abstract_text, current_part, previous_page, source_language, target_language, output_format
            )

            if translated_text == "context_length_exceeded":
                # Split the text in half and try again
                middle_index = len(current_part) // 2
                parts_to_translate.extend([current_part[:middle_index], current_part[middle_index:]])
            elif translated_text == "content_filter_triggered":
                result.append(f"\n***Content filter triggered on page {page_num + 1} - text skipped***\n")
            elif translated_text == '':
                result.append(f"\n***Translation error on page {page_num + 1}.***\n")
            else:
                result.append(translated_text)

        return f"\n\n-- Page {page_num + 1} -- \n\n" + "\n".join(result)

    def translate_document(self, pages: Iterable[PDFPage], abstract_text: Optional[str], page_nums_str: Optional[str],
                           source_language: str, target_language: str, output_file: Optional[str] = None, 
                           auto_save: bool = False, progressive_save: bool = False, input_file_path: Optional[str] = None) -> List[str]:
        """Translate all pages in a document."""
        from .file_output import FileOutputHandler
        
        document_text: list[str] = []
        start_page, end_page = extract_page_nums(page_nums_str)
        
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
        
        # Apply page range if specified
        if page_nums_str:
            pages = islice(pages, start_page, end_page + 1)

        # For progressive saving
        progressive_output_path = None

        page_text = ""
        for i, page in tqdm(enumerate(pages, start=start_page), desc="Translating... ", ascii=True):
            previous_page = page_text
            page_text = self.pdf_processor.process_page(page)
            
            try:
                translated_text = self.generate_text(
                    abstract_text or '', page_text, previous_page, i, source_language, target_language, output_format
                )
                document_text.append(translated_text)
                
                # Add delay between pages to prevent rate limiting and content filter triggers
                # This helps avoid jailbreak detection issues
                if i > start_page:  # Don't delay on the first page
                    time.sleep(PAGE_DELAY_SECONDS)
                
                # Save page progressively if requested
                if progressive_save and (output_file or auto_save):
                    is_first_page = (i == start_page)
                    progressive_output_path = FileOutputHandler.save_page_progressively(
                        translated_text, 
                        input_file_path,
                        output_file,
                        auto_save,
                        source_language,
                        target_language,
                        is_first_page
                    )
                    
            except Exception as e:
                error_message = f"\n***Translation error on page {i + 1}: {e}***\n"
                document_text.append(error_message)
                print(f"Error on page {i + 1}: {e}")
                
                # Still save the error message progressively
                if progressive_save and (output_file or auto_save):
                    is_first_page = (i == start_page and len(document_text) == 1)
                    FileOutputHandler.save_page_progressively(
                        error_message,
                        input_file_path,
                        output_file,
                        auto_save,
                        source_language,
                        target_language,
                        is_first_page
                    )
                
                # Continue with next page instead of stopping
                continue

        # If progressive saving was used, inform user about the output file
        if progressive_save and progressive_output_path:
            print(f"\nProgressive translation saved to: {progressive_output_path}")

        return document_text
    
    def translate_text_pages(self, text_pages: List[str], abstract_text: Optional[str],
                            source_language: str, target_language: str, output_file: Optional[str] = None, 
                            auto_save: bool = False, progressive_save: bool = False, input_file_path: Optional[str] = None) -> List[str]:
        """Translate pre-extracted text pages (e.g., from Word documents)."""
        from .file_output import FileOutputHandler
        
        document_text: list[str] = []
        
        # Determine output format based on whether file output is requested
        if output_file:
            if output_file.lower().endswith('.pdf'):
                output_format = "pdf"
            elif output_file.lower().endswith('.docx'):
                output_format = "docx"
            elif output_file.lower().endswith('.txt'):
                output_format = "txt"
            else:
                output_format = "file"
        elif auto_save:
            output_format = "txt"  # Auto-save defaults to txt format
        else:
            output_format = "console"

        # For progressive saving
        progressive_output_path = None

        previous_page = ""
        for i, page_text in tqdm(enumerate(text_pages), desc="Translating... ", ascii=True):
            try:
                translated_text = self.generate_text(
                    abstract_text or '', page_text, previous_page, i, source_language, target_language, output_format
                )
                document_text.append(translated_text)
                
                # Update previous page for context
                previous_page = page_text
                
                # Add delay between API calls (except for first page)
                if i > 0:  # Don't delay on the first page
                    time.sleep(PAGE_DELAY_SECONDS)
                
                # Save page progressively if requested
                if progressive_save and (output_file or auto_save):
                    is_first_page = (i == 0)
                    progressive_output_path = FileOutputHandler.save_page_progressively(
                        translated_text, 
                        input_file_path,
                        output_file,
                        auto_save,
                        source_language,
                        target_language,
                        is_first_page
                    )
                    
            except Exception as e:
                error_message = f"\n***Translation error on section {i + 1}: {e}***\n"
                document_text.append(error_message)
                print(f"Error on section {i + 1}: {e}")
                
                # Save page progressively if requested (even with error)
                if progressive_save and (output_file or auto_save):
                    is_first_page = (i == 0)
                    progressive_output_path = FileOutputHandler.save_page_progressively(
                        error_message, 
                        input_file_path,
                        output_file,
                        auto_save,
                        source_language,
                        target_language,
                        is_first_page
                    )

        # If progressive saving was used, print the final output path
        if progressive_save and progressive_output_path:
            print(f"\nProgressive saving completed. Final output: {progressive_output_path}")

        return document_text
    
    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of token usage."""
        return self.token_tracker.get_usage_summary()
    
    def print_usage_report(self):
        """Print a formatted usage report."""
        self.token_tracker.print_usage_report()
    
    def get_daily_usage(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get usage for a specific date or today."""
        return self.token_tracker.get_daily_usage(date)
    
    def update_model_pricing(self, model: str, input_price: float, output_price: float):
        """Update pricing for a specific model."""
        self.token_tracker.update_pricing(model, input_price, output_price)
