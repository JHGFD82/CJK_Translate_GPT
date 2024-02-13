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

