# Python standard libraries
import argparse
import json
import logging
import os
import re
from itertools import islice
from typing import List, Optional, Any, Iterator, Tuple, Union, BinaryIO, Callable

# Third-party libraries
from requests import Response, post
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects, RequestException
from tqdm import tqdm
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextContainer, LTTextBox, LTTextLine, LTFigure, LTChar, LTPage
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage


def validate_page_nums(value: str) -> str:
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
language = args.input_type
page_nums = validate_page_nums(args.page_nums) if args.page_nums else None
abstract = args.abstract
API_KEY = os.getenv('API_KEY')  # get API key from environment variables


def process_pdf(f: BinaryIO) -> Tuple[Iterator[PDFPage], PDFPageAggregator, PDFPageInterpreter]:
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pages = PDFPage.get_pages(f)

    return pages, device, interpreter


def extract_page_nums() -> Tuple[int, int]:
    """Extracts the start and end page numbers from the given string."""

    if page_nums is None:
        start_page = 1
        end_page = None
    elif '-' in page_nums:
        start_page, end_page = map(int, page_nums.split('-'))
    elif int(page_nums) > 0:
        start_page = int(page_nums)
        end_page = start_page
    else:
        raise ValueError(f"{page_nums} is not a valid page number.")

    if end_page is not None:
        end_page -= 1
    return start_page - 1, end_page


def parse_layout(layout: LTPage) -> str:
    """Function to parse the layout tree."""
    result = []
    stack = list(layout)  # Using a list as a stack

    while stack:
        lt_obj = stack.pop(0)
        if isinstance(lt_obj, (LTTextLine, LTChar, LTTextContainer)):
            result.append(lt_obj.get_text())
        elif isinstance(lt_obj, (LTFigure, LTTextBox)):
            stack.extend(list(lt_obj))  # Add children to the stack

    return "".join(result)

def generate_text(abstract_text: str, page_text: str, previous_page: str, i: int) -> str:
    result = []
    parts_to_translate = [page_text]

    while parts_to_translate:
        current_part = parts_to_translate.pop()
        translated_text = translate_page_text(abstract_text, current_part, previous_page)

        if translated_text == "context_length_exceeded":
            middle_index = len(current_part) // 2
            parts_to_translate.extend([current_part[:middle_index], current_part[middle_index:]])
        elif translated_text is None:
            result.append(f"\n***Translation error on page {i + 1}.***\n")
        else:
            result.append(translated_text)

    returned_text = f"\n\n-- Page {i + 1} -- \n\n" + "\n".join(result)

    return returned_text


def translate_document(pages: Iterator[PDFPage], interpreter: Any,
                       device: PDFPageAggregator, abstract_text: Optional[str]) -> List[str]:
    document_text = []
    start_page, end_page = extract_page_nums()
    pages = islice(pages, start_page, end_page + 1 if end_page is not None else None)
    page_text = ""
    for i, page in tqdm(enumerate(pages, start=start_page), desc="Translating... ", ascii=True):
        interpreter.process_page(page)
        layout = device.get_result()
        previous_page = page_text
        page_text = parse_layout(layout)
        translated_text = generate_text(abstract_text, page_text, previous_page, i)
        document_text.append(translated_text)

    return document_text


def main() -> None:
    if file:
        abstract_text = input('Enter abstract text: ') if abstract else None
        with open(file, 'rb') as f:
            pages, device, interpreter = process_pdf(f)
            document_text = translate_document(pages, interpreter, device, abstract_text)
        print("".join(document_text))
    elif custom_text:
        text_input = input('Enter custom text to be translated: ')
        translated_text = generate_text('', text_input, '', 0)
        print(translated_text)


if __name__ == '__main__':
    main()