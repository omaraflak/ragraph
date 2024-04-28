from abc import ABC, abstractmethod

Embedding = list[float]


class Model(ABC):
    @abstractmethod
    def generate_embeddings(self, texts: list[str]) -> list[Embedding]:
        raise NotImplementedError()


    @abstractmethod
    def generate_texts(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError()
