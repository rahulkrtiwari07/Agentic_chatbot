from langchain_community.document_transformers import DoctranTextTranslator
import langdetect
import os 
from langchain_core.documents import Document
from dotenv import load_dotenv
import asyncio
import json

load_dotenv('.env.example')

# Constants for language codes
HINDI = "hi"
ENGLISH = "en"

api_key = os.getenv('OPENAI_API_KEY')

def detect_language(text):
    """Detects the language of the given text.

    Args:
        text: The text to detect the language of.

    Returns:
        The detected language code (e.g., 'hi', 'en').
    """

    try:
        return langdetect.detect(text)
    except langdetect.LangDetectException:
        # Handle language detection errors (e.g., log, return default language)
        return None

async def translate(text, target_language):
    """Translates the given text to the specified target language.

    Args:
        text: The text to translate.
        target_language: The target language code (e.g., 'hi', 'en').

    Returns:
        The translated text.
    """

    documents = [Document(page_content=text)]
    translator = DoctranTextTranslator(
        openai_api_model="gpt-3.5-turbo",
        language=target_language,
        openai_api_key=api_key
    )
    translated_text = await translator.atransform_documents(documents)
    return json.dumps(translated_text[0].page_content)

async def main():
    query = "राष्ट्रीय परिवार स्वास्थ्य सर्वेक्षण 2019-21 के अनुसार कितने प्रतिशत परिवारों ने पीने के पानी के बेहतर स्रोत का उपयोग किया?"

    detected_language = detect_language(query)
    if detected_language == HINDI:
        english_query = await translate(query, ENGLISH)
        print(english_query)
        print(type(english_query))
        # Consider translating back to Hindi directly if needed
        # hindi_query = await translate(english_query, HINDI)
        # print(hindi_query)

if __name__ == "__main__":
    asyncio.run(main())