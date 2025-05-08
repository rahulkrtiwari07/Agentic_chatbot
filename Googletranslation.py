from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from google.api_core.protobuf_helpers import get_messages
from google.cloud import translate_v2 as translate
import langdetect
import os
from dotenv import load_dotenv
import asyncio

load_dotenv('.env.example')

# Constants for language codes
HINDI = "hi"
ENGLISH = "en"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'pfi-chatbot-20899faecc30.json'

def detect_language(text):
        client = translate.Client()
        result = client.detect_language(text)
        return result['language']

def translate_to_english(text):
        client = translate.Client()
        result = client.translate(text, target_language='en')
        return result['translatedText']

def translate_to_hindi(text):
      client = translate.Client()
      result = client.translate(text, target_language='hi')
      return result['translatedText']
      
async def main():
    query = "राष्ट्रीय परिवार स्वास्थ्य सर्वेक्षण 2019-21 के अनुसार कितने प्रतिशत परिवारों ने पीने के पानी के बेहतर स्रोत का उपयोग किया?"

    detected_language = detect_language(query)
    if detected_language == HINDI:
        english_query = translate_to_english(query)
        print(english_query)

if __name__ == "__main__":
    asyncio.run(main())