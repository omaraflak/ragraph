import prompts
from model import Model, Embedding
from dataclasses import dataclass


@dataclass
class QuestionAnswer:
    question: str
    answer: str
    context: str


@dataclass
class Scores:
    groundedness: float
    relevance: float
    standalone: float


@dataclass
class Entry:
    text: str
    embedding: Embedding


@dataclass
class Dataset:
    entries: list[Entry]


def load_raw_data() -> list[str]:
    pass


def make_dataset(model: Model, chunks: list[str]) -> Dataset:
    pass


def generate_questions_answers(model: Model, dataset: list[str]) -> list[QuestionAnswer]:
    pass


def calculate_scores(model: Model, questions: list[QuestionAnswer]) -> list[Scores]:
    pass


def retrieve(model: Model, question: str, dataset: Dataset, top_k: int) -> list[Entry]:
    pass


def answer_questions(model: Model, questions: list[QuestionAnswer], dataset: Dataset, top_k: int) -> list[str]:
    pass


# def evaluate_answers(model: Model, questions)