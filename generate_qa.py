import json
import prompts
from model import Model
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin


@dataclass
class QuestionAnswer(DataClassJsonMixin):
    question: str
    answer: str
    context: str


@dataclass
class Scores(DataClassJsonMixin):
    groundedness: float
    standalone: float


@dataclass
class QuestionAnswerScore(DataClassJsonMixin):
    question_answer: QuestionAnswer
    scores: Scores


@dataclass
class QuestionAnswerDataset(DataClassJsonMixin):
    items: list[QuestionAnswerScore]


def load_data_chunks(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        return f.read().split('.')


def extract_key_values(text: str, keys: list[str]) -> dict[str, str]:
    results = {key: '' for key in keys}
    for line in text.splitlines():
        values = line.split(':')
        if len(values) == 2 and values[0] in keys:
            results[values[0]] = values[1].strip()
    return results


def generate_questions_answers(model: Model, chunks: list[str]) -> list[QuestionAnswer]:
    requests = [prompts.question_answer_generation.format(context=chunk) for chunk in chunks]
    results: list[QuestionAnswer] = []
    for context, response in zip(chunks, model.generate_texts(requests)):
        values = extract_key_values(response, ['question', 'answer'])
        results.append(QuestionAnswer(values['question'], values['answer'], context))
    return results


def calculate_scores(model: Model, questions_answers: list[QuestionAnswer]) -> list[Scores]:
    requests = [
        prompt
        for question_answer in questions_answers
        for prompt in [
            prompts.groundedness_scoring.format(question=question_answer.question, context=question_answer.context),
            prompts.standalone_scoring.format(question=question_answer.question),
        ]
    ]
    responses = model.generate_texts(requests)
    return [
        Scores(
            float(extract_key_values(responses[i], ['rating'])['rating'] or '1'),
            float(extract_key_values(responses[i + 1], ['rating'])['rating'] or '1'),
        )
        for i in range(0, len(responses), 2)
    ]


def generate_dataset(model: Model, chunks: list[str]) -> QuestionAnswerDataset:
    questions_answers = generate_questions_answers(model, chunks)
    scores = calculate_scores(model, questions_answers)
    items = [
        QuestionAnswerScore(question_answer, score)
        for question_answer, score in zip(questions_answers, scores)
        if score.groundedness >= 4 and score.standalone >= 4
    ]
    return QuestionAnswerDataset(items)


def main():
    model = Model()
    chunks = load_data_chunks('source.txt')
    dataset = generate_dataset(model, chunks)
    with open('dataset.json', 'w') as f:
        json.dump(dataset.to_dict(), f, indent=2)


main()