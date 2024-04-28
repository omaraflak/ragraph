import os
import dotenv
import requests
from model import Model, Embedding
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor


class Gemini(Model):
    def __init__(self):
        dotenv.load_dotenv()
        self._google_api_key = os.environ['GOOGLE_API_KEY']
        genai.configure(api_key=self._google_api_key)
        self.gemini = genai.GenerativeModel('gemini-pro')


    def generate_embeddings(self, texts: list[str]) -> list[Embedding]:
        model = 'models/text-embedding-004'
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:batchEmbedContents?key={self._google_api_key}"
        json = {
            'requests': [
                {
                    'model': model,
                    'content': {
                        'parts': [{'text': text}],
                        'role': 'user'   
                    }
                }
                for text in texts
            ]
        }
        result = requests.post(url, json=json).json()
        return [x['values'] for x in result['embeddings']]


    def generate_texts(self, prompts: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=10) as executor:
            return [
                x.text
                for x in executor.map(self.gemini.generate_content, prompts)
            ]
