# Install the Portkey AI Gateway and Mimetypes packages with pip
#   pip -i portkey-ai mimetypes
# 
# For more information on the SDK see https://portkey.ai/docs/api-reference/sdk/python
# 
from portkey_ai import Portkey
import base64
from mimetypes import guess_type

from dotenv import load_dotenv
from .utils import get_api_key

# Load environment variables from a .env file
load_dotenv()

# Before executing this code, define the API Key within an enironment variable in your OS
# Linux BASH example: export PORTKEY_API_KEY=<key provided to you>

# Import API key from OS environment variables
api_key, professor_name = get_api_key("heller")

client = Portkey(api_key=api_key)


# Base 64 encode local image and return text to be included in AI prompt
def local_image_to_data_url(image_path: str):
    """
    Get the url of a local image
    """
    mime_type, _ = guess_type(image_path)

    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_encoded_data}"

# This function will submit a simple prompt and image file to the chosen model
def image_prompt_example(model_to_be_used: str, image_file: str):
    # Establish a connection to your Azure OpenAI instance
    
    try:
        response = client.chat.completions.create( # type: ignore[misc]
        model="gpt-5", 
        # Prompt parameters may also be defined, depending on model capabilities
        #temperature=0.5, # temperature = how creative/random the model is in generating response - 0 to 1 with 1 being most creative
        #max_tokens=1000, # max_tokens = token limit on context to send to the model
        #top_p=0.5, # top_p = diversity of generated text by the model considering probability attached to token - 0 to 1 - ex. top_p of 0.1 = only tokens within the top 10% probability are considered
        messages=[
        {"role": "system", "content": "You are an expert OCR assistant. Extract all visible text from the provided image exactly as it appears. The text may be in vertical or horizontal orientation, including East Asian languages (Chinese, Japanese, Korean). Return ONLY the extracted text without any commentary, notes, disclaimers, or translation accuracy warnings."}, # describes model identity and purpose
        {"role": "user", "content": [{"type": "text", "text": "What text is in this image?"}, {"type": "image_url", "image_url": {"url": local_image_to_data_url(image_file)}}]}, # user prompt
               ]
        )
        print("\n"+response.choices[0].message.content)

    except Exception as e:
        print(e.message)

# Execute the example functions
#
if __name__ == "__main__":

    image_file = "tests/essays-on-the-history-philosophy-and-religion-of-the-chinese-image-HBX848.jpg" # example image file path

    # Test image analysis ONLY with those models that support images
    
    print("\nModel: gpt-5")
    image_prompt_example("gpt-5", image_file)
