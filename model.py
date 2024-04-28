import os
import dotenv
import requests
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
# from transformers import AutoModelForCausalLM, AutoTokenizer

Embedding = list[float]


class Model:
    def __init__(self):
        dotenv.load_dotenv()
        self._google_api_key = os.environ['GOOGLE_API_KEY']

        genai.configure(api_key=self._google_api_key)
        self.gemini = genai.GenerativeModel('gemini-pro')

        # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
        # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side="left")
        # tokenizer.pad_token = tokenizer.eos_token
        # device = "cpu"


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


    def _generate_embedding(self, text: str) -> Embedding:
        return self.generate_embeddings([text])[0]


    def generate_texts(self, prompts: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=10) as executor:
            return [
                x.text
                for x in executor.map(self.gemini.generate_content, prompts)
            ]
        # model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        # generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
        # decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # return decoded[0]