import os
import dotenv
import requests
from model import Model, Embedding
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


class HuggingFace(Model):
    def __init__(self):
        dotenv.load_dotenv()
        hf_token = os.environ['HF_TOKEN']
        self.embedding_model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda"


    def generate_embeddings(self, texts: list[str]) -> list[Embedding]:
        return [x.tolist() for x in self.embedding_model.encode(texts)]


    def generate_texts(self, prompts: list[str]) -> list[str]:
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0]
