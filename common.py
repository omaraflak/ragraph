import numpy as np
from model import Model, Embedding
from gemini import Gemini
from hf import HuggingFace


def get_model(name: str) -> Model:
    if name == 'gemini':
        return Gemini()
    if name == 'hf':
        return HuggingFace()

    raise NotImplementedError(f"No model named '{name}'")


def read_data_chunks(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        return [x.strip() for x in f.read().split('.') if x.strip()]


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    return np.dot(a, np.transpose(b)) / (np.linalg.norm(a) * np.linalg.norm(b))
