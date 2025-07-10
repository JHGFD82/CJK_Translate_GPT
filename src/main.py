'''This script extracts text from a PDF file and translates it between different languages using the GPT engine. 
It supports translation between Chinese, Japanese, Korean, and English in any direction.
This script is for Princeton University use only and will only function using a valid API key to the AI Sandbox.'''

# Python standard libraries
import argparse
import logging
import os
import re
from itertools import islice
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


parser = argparse.ArgumentParser(description='Extract text from PDF and translate it using the GPT engine.')

input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('-C', '--Chinese', dest='input_type', action='store_const', const='Chinese',
                         help='Input is Chinese text')
input_group.add_argument('-J', '--Japanese', dest='input_type', action='store_const', const='Japanese',
                         help='Input is Japanese text')
input_group.add_argument('-K', '--Korean', dest='input_type', action='store_const', const='Korean',
                         help='Input is Korean text')

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

args = parser.parse_args()

# Set up global variables for script
file = args.input_PDF
custom_text = args.custom_text
page_nums = validate_page_nums(args.page_nums) if args.page_nums else None
abstract = args.abstract

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


def main() -> None:
    if file:
        abstract_text = input('Enter abstract text: ') if abstract else None
        try:
            with open(file, 'rb') as f:
                pages, device, interpreter = process_pdf(f)
                document_text = translate_document(pages, interpreter, device, abstract_text, page_nums, language, target_language)
            print("".join(document_text))
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


if __name__ == '__main__':
    main()
