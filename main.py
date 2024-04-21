import os
import pickle
import dotenv
import requests
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer


Embedding = list[float]


dotenv.load_dotenv()
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel('gemini-pro')

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
device = "cpu" # the device to load the model onto


@dataclass
class Node:
    text: str
    embedding: Embedding
    neighbors: list[tuple['Node', Embedding]]


    def __hash__(self) -> int:
        return int(np.prod(self.embedding))


def generate_embeddings(texts: list[str]) -> list[Embedding]:
    model = 'models/text-embedding-004'
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:batchEmbedContents?key={GOOGLE_API_KEY}"
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


def _generate_embedding(text: str) -> Embedding:
    return generate_embeddings([text])[0]


def generate_texts(prompts: list[str]) -> list[str]:
    model_inputs = tokenizer(
         prompts, return_tensors="pt", padding=True
    ).to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0]


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    return np.dot(a, np.transpose(b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_edges(pairs: list[tuple[str, str]]) -> list[Embedding]:
    prompt = """I will give you two chunks of texts. Output another piece of text (maximum 100 words) to represent the semantic relationship between the two chuncks of text.
The generated text must be simple, objective, and should not provide any extra context that is not provided by the input texts.
Please take into consideration that sometimes the two chunks may not have any semantic or logic relationship; In that case, please anser that the two chunchs have no relationship, and avoid at all cost to create a imaginary/erronous connection or relationship.
However if you think the two chuncks have a distant relationship, feel free to briefly mention this relationship.

Example:
chunk 1: E=m*c^2 is the mass energy physic equation.
chunk 2: Albert Einstein wrote the mass energy physic equation.
relationship: Albert Einstein wrote the mass energy physic equation E=m*c^2.

Example:
chunk 1: The speed of light is 3x10^8 metre per second.
chunk 2: I love bananas.
relationship: There is no relationship. 

Example:
chunk 1: Barack Obama was the 44th president of the United States.
chunk 2: Donald Trump was the 45th president of the United States.
relationship: Barack Obama and Donald Trump were both United States presidents, Donald Trump was Barack Obama successor. 

Now it's your turn:
chunk 1: {}
chunk 2: {}
relationship: """
    requests = [prompt.format(a, b) for a, b in pairs]
    return generate_embeddings(generate_texts(requests))


def create_nodes(chunks: list[str]) -> list[Node]:
    nodes = [
        Node(chunk, embedding, [])
        for chunk, embedding in zip(chunks, generate_embeddings(chunks))
    ]
    edges = [
        (a, b, (a.text, b.text))
        for a in nodes
        for b in nodes
        if a != b
    ]
    pairs = [pair for _, _, pair in edges]

    for (a, b, _), edge_embedding in zip(edges, compute_edges(pairs)):
        a.neighbors.append((b, edge_embedding))
        b.neighbors.append((a, edge_embedding))

    return nodes


def _retrieve_dfs(root: Node, query_embedding: Embedding, max_depth: int, min_similarity: float) -> list[Node]:
    result: set[Node] = set()
    queue = [(root, 0)]
    while queue:
        node, depth = queue.pop()
        if depth == max_depth:
            continue

        result.add(node)

        neigbors_scores = sorted([
            (neighbor, cosine_similarity(edge_embedding, query_embedding))
            for neighbor, edge_embedding in node.neighbors
        ], key=lambda x: x[1], reverse=True)
        candidates = [
            (neighbor, depth + 1)
            for neighbor, score in neigbors_scores
            if score >= min_similarity
        ]
        candidates = [(x[0], depth + 1) for neighb in neigbors_scores if x[1] >= min_similarity]

    queue.extend(candidates)
    
    return list(result)


def retrieve(nodes: list[Node], query: str, top_k: int, max_depth: int, min_similarity: float) -> list[Node]:
    query_embedding = _generate_embedding(query)
    similar_nodes = [(node, cosine_similarity(node.embedding, query_embedding)) for node in nodes]
    similar_nodes.sort(key=lambda x: x[1], reverse=True)
    similar_nodes = similar_nodes[:top_k]
    return list({
        x
        for node, _ in similar_nodes
        for x in _retrieve_dfs(node, query_embedding, max_depth, min_similarity)
    })


def save_graph(nodes: list['Node'], filename: str):
    with open(filename, 'wb') as file:
        pickle.dump(nodes, file)


def load_graph(filename: str) -> list['Node']:
    with open(filename, 'rb') as file:
        return pickle.load(file)


def main():
    # create graph
    chunks = [x.strip() for x in open('source.txt', 'r').read().split('.') if x.strip()]
    nodes = create_nodes(chunks)
    save_graph(nodes, 'graph.bin')

    # query graph
    query = 'thought police'
    results = retrieve(nodes, query, top_k=3, max_depth=4, min_similarity=0.3)
    for node in results:
        print(node.text)


main()