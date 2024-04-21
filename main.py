import os
import pickle
import dotenv
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai


Embedding = list[float]


dotenv.load_dotenv()
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
gemini = genai.GenerativeModel('gemini-pro')


@dataclass
class Node:
    text: str
    embedding: Embedding
    neighbors: list[tuple['Node', Embedding]]


    def __hash__(self) -> int:
        return int(np.prod(self.embedding))


def generate_embeddings(texts: list[str]) -> list[Embedding]:
    return [x.tolist() for x in model.encode(texts)]


def _generate_embedding(text: str) -> Embedding:
    return generate_embeddings([text])[0]


def generate_texts(prompts: list[str]) -> list[str]:
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(gemini.generate_content, prompts))
        return [result.text for result in results]


def cosine_similarity(a: Embedding, b: Embedding) -> list:
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

        candidates = [x[0] for x in neigbors_scores if x[1] >= min_similarity]

        queue.extend(candidates)
    
    return list(result)


def retrieve(nodes: list[Node], query: str, top_k: int, max_depth: int, min_similarity: float) -> list[Node]:
    query_embedding = _generate_embedding(query)
    similar_nodes = [(node, cosine_similarity(node.embedding, query_embedding)) for node in nodes]
    similar_nodes.sort(key=lambda x: x[1], reverse=True)
    similar_nodes = similar_nodes[:top_k]
    return list({
        x
        for node in similar_nodes
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


# main()
# print(_generate_embedding('hello'))

for x in generate_texts(['my name is ', 'some colors are: yellow, red, ']):
    print(x)
