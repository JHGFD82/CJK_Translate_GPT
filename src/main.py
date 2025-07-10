'''This script extracts text from a PDF file and translates it between different languages using the GPT engine. 
It supports translation between Chinese, Japanese, Korean, and English in any direction.
This script is for Princeton University use only and will only function using a valid API key to the AI Sandbox.'''

# Python standard libraries
import argparse
import logging
import os
import re
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import List, Optional, Any, Iterator, Tuple, BinaryIO

# Third-party libraries
from openai import AzureOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextContainer, LTTextBox, LTTextLine, LTFigure, LTChar, LTPage
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

load_dotenv()

# Import API key from OS environment variables
try:
    sandbox_api_key = os.environ['AI_SANDBOX_KEY']
except KeyError:
    print("Error: AI_SANDBOX_KEY environment variable not found.")
    print("Please set your API key in the environment variables.")
    exit(1)

def validate_page_nums(value: str) -> str:
    """Validate the page numbers input."""
    if not re.match(r"^\d+(-\d+)?$", value):
        raise argparse.ArgumentTypeError("Letters, commas, and other symbols not allowed.")
    return value


def parse_language_code(value: str) -> tuple[str, str]:
    """Parse language code like 'CE' into source and target languages."""
    if len(value) != 2:
        raise argparse.ArgumentTypeError("Language code must be exactly 2 characters (e.g., CE, JK, etc.)")
    
    language_map = {
        'C': 'Chinese',
        'J': 'Japanese', 
        'K': 'Korean',
        'E': 'English'
    }
    
    source_char = value[0].upper()
    target_char = value[1].upper()
    
    if source_char not in language_map:
        raise argparse.ArgumentTypeError(f"Invalid source language code '{source_char}'. Use C, J, K, or E.")
    if target_char not in language_map:
        raise argparse.ArgumentTypeError(f"Invalid target language code '{target_char}'. Use C, J, K, or E.")
    if source_char == target_char:
        raise argparse.ArgumentTypeError("Source and target languages cannot be the same.")
    
    return language_map[source_char], language_map[target_char]


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
''')

# Language selection - single argument combining source and target
parser.add_argument('language_code', type=parse_language_code, metavar='LANG_CODE',
                    help='Two-letter language code: first letter is source, second is target (e.g., CE, JK, EJ)')

source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument('-i', '--input_PDF', dest='input_PDF', type=str,
                          help='The name of the input PDF file')
source_group.add_argument('-c', '--custom_text', dest='custom_text', action='store_true',
                          help='Input custom text to be translated')

parser.add_argument('-p', '--page_nums', dest='page_nums', type=str,
                    help='Page numbers to output\nEnter either a single page number or a range in this format: '
                         '[starting page number]-[ending page number]\nNo spaces, letters, commas or other symbols '
                         'are allowed')

parser.add_argument('-a', '--abstract', dest='abstract', action='store_true',
                    help='The text has an abstract')

parser.add_argument('-o', '--output', dest='output_file', type=str,
                    help='Output file path to save the translation (with .txt or .pdf extension)')

parser.add_argument('--auto-save', dest='auto_save', action='store_true',
                    help='Automatically save output to a timestamped file in the same directory as input PDF')

args = parser.parse_args()

# Extract source and target languages from the language code
language, target_language = args.language_code

# Set up global variables for script
file = args.input_PDF
custom_text = args.custom_text
page_nums = validate_page_nums(args.page_nums) if args.page_nums else None
abstract = args.abstract
output_file = args.output_file
auto_save = args.auto_save

# Set the model deployment name that the prompt should be sent to
available_models = [
                    "o3-mini", 
                    "gpt-4o-mini", 
                    "gpt-4o", 
                    "gpt-35-turbo-16k", 
                    "Meta-Llama-3-1-70B-Instruct-htzs", 
                    "Meta-Llama-3-1-8B-Instruct-nwxcg", 
                    "Mistral-small-zgjes"
                ]

# Set up logging once
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_pdf(f: BinaryIO) -> Tuple[Iterator[PDFPage], PDFPageAggregator, PDFPageInterpreter]:
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pages = PDFPage.get_pages(f)

    return pages, device, interpreter


def extract_page_nums(page_nums_str: Optional[str]) -> Tuple[int, int]:
    """Extracts the start and end page numbers from the given string."""
    end_page: int = 0
    if page_nums_str is None:
        start_page = 1
        end_page = 0
    elif '-' in page_nums_str:
        start_page, end_page = map(int, page_nums_str.split('-'))
    elif int(page_nums_str) > 0:
        start_page = int(page_nums_str)
        end_page = start_page
    else:
        raise ValueError(f"{page_nums_str} is not a valid page number.")

    if end_page != 0:
        end_page -= 1
    return start_page - 1, end_page


def parse_layout(layout: LTPage) -> str:
    """Function to parse the layout tree."""
    result: list[str] = []
    stack = list(layout)  # Using a list as a stack

    while stack:
        lt_obj = stack.pop(0)
        if isinstance(lt_obj, (LTTextLine, LTChar, LTTextContainer)):
            result.append(lt_obj.get_text())
        elif isinstance(lt_obj, (LTFigure, LTTextBox)):
            stack.extend(list(lt_obj))  # Add children to the stack

    return "".join(result)


def generate_process_text(abstract_text: str, page_text: str, previous_page: str) -> str:
    context = abstract_text if abstract_text else previous_page[int(len(previous_page) * .65):]
    if context:
        context = f"--Context: \n{context}"
    return f"--Current Page: \n{page_text}\n{context}"


def translate_text(text: str, model_to_be_used: str, source_language: str, target_language: str) -> str:
    """Translate text using the specified model."""
    sandbox_api_version = "2025-03-01-preview"
    sandbox_endpoint: str = "https://api-ai-sandbox.princeton.edu/"

    client = AzureOpenAI(
        api_key=sandbox_api_key,
        azure_endpoint=sandbox_endpoint,
        api_version=sandbox_api_version  # current api version not in preview
    )

    prompt_system = (f'Follow the instructions carefully. Please act as a professional translator from {source_language} '
                     f'to {target_language}. I will provide you with text from a PDF document, and your task is '
                     f'to translate it from {source_language} to {target_language}. Please only output the translation and do not '
                     f'output any irrelevant content. If there are garbled characters or other non-standard text '
                     f'content, delete the garbled characters. '
                     f'You can format and line break the output yourself using "\\n" for line breaks. '
                     f'You may be provided with "--Context: " and the text from either the document\'s abstract or '
                     f'a sample of text from the previous page. You will also be provided with "--Current Page: " '
                     f'which includes the OCR characters of the current page. Only output the {target_language} translation of '
                     f'the "--Current Page: ". Do not output the context, nor the "--Context: " and "--Current Page: " '
                     f'labels.')
    prompt_user = (f'Translate only the {source_language} text of the "--Current Page: " to {target_language}, without outputting any other '
                   f'content, and without outputting anything related to "--Context: ", if provided. Do not provide '
                   f'any prompts to the user, for example: "This is the translation of the current page.":\n') + text
 
    try:
        logging.info(f'Making API call to model: {model_to_be_used}')
        response = client.chat.completions.create(
            model=model_to_be_used,
            temperature=0.5, # temperature = how creative/random the model is in generating response - 0 to 1 with 1 being most creative
            max_tokens=1000, # max_tokens = token limit on context to send to the model
            top_p=0.5, # top_p = diversity of generated text by the model considering probability attached to token - 0 to 1 - ex. top_p of 0.1 = only tokens within the top 10% probability are considered
            messages=[
                {"role": "system", "content": prompt_system}, # describes model identity and purpose
                {"role": "user", "content": prompt_user}, # user prompt
            ]
        )
        
        # Log response details
        logging.info(f'API call successful. Response ID: {response.id}')
        logging.info(f'Model used: {response.model}')
        
        # Log token usage if available
        if response.usage:
            logging.info(f'Tokens used - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}')
        else:
            logging.info('Token usage information not available')
        
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
        error_type = type(e).__name__
        error_message = str(e)
        
        # Check for specific OpenAI error types
        if "context_length_exceeded" in error_message.lower() or "maximum context length" in error_message.lower():
            logging.error(f'Context length exceeded: {error_message}')
            return "context_length_exceeded"
        elif "rate_limit" in error_message.lower() or "rate limit" in error_message.lower():
            logging.error(f'Rate limit exceeded: {error_message}')
            raise Exception(f'Rate limit exceeded: {error_message}')
        elif "invalid_request" in error_message.lower():
            logging.error(f'Invalid request: {error_message}')
            raise Exception(f'Invalid request: {error_message}')
        elif "authentication" in error_message.lower() or "unauthorized" in error_message.lower():
            logging.error(f'Authentication error: {error_message}')
            raise Exception(f'Authentication error: {error_message}')
        else:
            logging.error(f'API call failed with {error_type}: {error_message}')
            raise Exception(f'API call failed with {error_type}: {error_message}')


def translate_page_text(abstract_text: str, page_text: str, previous_page: str, source_language: str, target_language: str) -> str:
    """Translate page text with context."""
    process_text = generate_process_text(abstract_text, page_text, previous_page)
    # Use GPT-4o by default, but make it configurable
    default_model = "gpt-4o"
    if default_model not in available_models:
        default_model = available_models[0]  # Fallback to first available model
    translated_text = translate_text(process_text, default_model, source_language, target_language)
    return translated_text


def generate_text(abstract_text: str, page_text: str, previous_page: str, i: int, source_language: str, target_language: str) -> str:
    """Generate translated text for a page, handling context length limits."""
    result: list[str] = []
    parts_to_translate = [page_text]

    while parts_to_translate:
        current_part = parts_to_translate.pop()
        translated_text: str = translate_page_text(abstract_text, current_part, previous_page, source_language, target_language)

        if translated_text == "context_length_exceeded":
            middle_index = len(current_part) // 2
            parts_to_translate.extend([current_part[:middle_index], current_part[middle_index:]])
        elif translated_text == '':
            result.append(f"\n***Translation error on page {i + 1}.***\n")
        else:
            result.append(translated_text)

    returned_text = f"\n\n-- Page {i + 1} -- \n\n" + "\n".join(result)

    return returned_text


def translate_document(pages: Iterator[PDFPage], interpreter: Any,
                       device: PDFPageAggregator, abstract_text: Optional[str], page_nums_str: Optional[str], 
                       source_language: str, target_language: str) -> List[str]:
    """Translate all pages in a document."""
    document_text: list[str] = []
    start_page, end_page = extract_page_nums(page_nums_str)
    pages = islice(pages, start_page, end_page + 1 if end_page != 0 else None)
    page_text = ""
    for i, page in tqdm(enumerate(pages, start=start_page), desc="Translating... ", ascii=True):
        interpreter.process_page(page)
        layout = device.get_result()
        previous_page = page_text
        page_text = parse_layout(layout)
        translated_text = generate_text(abstract_text or '', page_text, previous_page, i, source_language, target_language)
        document_text.append(translated_text)

    return document_text


def save_to_text_file(content: str, output_path: str) -> None:
    """Save content to a text file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logging.info(f'Translation saved to text file: {output_path}')
        print(f"\nTranslation saved to: {output_path}")
    except Exception as e:
        logging.error(f'Error saving to text file: {e}')
        print(f"Error saving to text file: {e}")


def save_to_pdf(content: str, output_path: str) -> None:
    """Save content to a PDF file using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter, 
                              rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        # Create story (content container)
        from reportlab.platypus import Flowable
        story: list[Flowable] = []
        styles = getSampleStyleSheet()
        
        # Try to register fonts that support CJK characters
        try:
            # This is a fallback - users might need to install appropriate fonts
            normal_style = ParagraphStyle(
                'Normal',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=12,
                spaceAfter=12,
                encoding='utf-8'
            )
        except:
            normal_style = styles['Normal']
        
        # Split content into paragraphs and add to story
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Clean up the text for PDF
                clean_text = para.strip()
                try:
                    p = Paragraph(clean_text, normal_style)
                    story.append(p)
                    story.append(Spacer(1, 12))
                except:
                    # Fallback for problematic characters
                    clean_text = clean_text.encode('ascii', 'ignore').decode('ascii')
                    p = Paragraph(clean_text, normal_style)
                    story.append(p)
                    story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        logging.info(f'Translation saved to PDF file: {output_path}')
        print(f"\nTranslation saved to PDF: {output_path}")
        
    except ImportError:
        logging.warning('reportlab not installed. Falling back to text file.')
        print("Warning: reportlab not installed. Saving as text file instead.")
        text_output_path = output_path.replace('.pdf', '.txt')
        save_to_text_file(content, text_output_path)
    except Exception as e:
        logging.error(f'Error saving to PDF: {e}')
        print(f"Error saving to PDF: {e}")
        print("Falling back to text file...")
        text_output_path = output_path.replace('.pdf', '.txt')
        save_to_text_file(content, text_output_path)


def generate_output_filename(input_file: str, source_lang: str, target_lang: str, extension: str = '.txt') -> str:
    """Generate an output filename based on input file and languages."""
    input_path = Path(input_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{input_path.stem}_{source_lang}to{target_lang}_{timestamp}{extension}"
    return str(input_path.parent / output_name)


def save_translation_output(content: str, input_file: str, output_file: str, auto_save: bool, 
                          source_lang: str, target_lang: str) -> None:
    """Save translation output to file based on user preferences."""
    if not content.strip():
        print("No content to save.")
        return
    
    # Determine output file path
    if output_file:
        output_path = output_file
    elif auto_save and input_file:
        # Auto-generate filename with timestamp
        output_path = generate_output_filename(input_file, source_lang, target_lang, '.txt')
    else:
        # No saving requested
        return
    
    # Ensure directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine file type and save accordingly
    if output_path.lower().endswith('.pdf'):
        save_to_pdf(content, output_path)
    else:
        # Default to text file
        if not output_path.lower().endswith('.txt'):
            output_path += '.txt'
        save_to_text_file(content, output_path)


def main() -> None:
    if file:
        abstract_text = input('Enter abstract text: ') if abstract else None
        try:
            with open(file, 'rb') as f:
                pages, device, interpreter = process_pdf(f)
                document_text = translate_document(pages, interpreter, device, abstract_text, page_nums, language, target_language)
            
            # Join all translated content
            full_translation = "".join(document_text)
            
            # Display the translation
            print(full_translation)
            
            # Save the translation if requested
            if output_file or auto_save:
                save_translation_output(full_translation, file, output_file, auto_save, language, target_language)
            
        except FileNotFoundError:
            print(f"Error: File '{file}' not found.")
            exit(1)
        except Exception as e:
            print(f"Error processing PDF: {e}")
            exit(1)
    elif custom_text:
        text_input = input('Enter custom text to be translated: ')
        translated_text = generate_text('', text_input, '', 0, language, target_language)
        print(translated_text)
        
        # Save custom text translation if requested
        if output_file or auto_save:
            # For custom text, use a generic filename
            input_filename = "custom_text_translation"
            save_translation_output(translated_text, input_filename, output_file, auto_save, language, target_language)


if __name__ == '__main__':
    main()
