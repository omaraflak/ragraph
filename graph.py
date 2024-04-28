import pickle
import prompts
import numpy as np
from dataclasses import dataclass
from model import Model, Embedding


model = Model()


@dataclass
class Node:
    text: str
    embedding: Embedding
    neighbors: list[tuple['Node', Embedding]]


    def __hash__(self) -> int:
        return int(np.prod(self.embedding))


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    return np.dot(a, np.transpose(b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_edges(pairs: list[tuple[str, str]]) -> list[Embedding]:
    requests = [prompts.chunks_relationship.format(chunk1=a, chunk2=b) for a, b in pairs]
    return model.generate_embeddings(model.generate_texts(requests))


def create_nodes(chunks: list[str]) -> list[Node]:
    nodes = [
        Node(chunk, embedding, [])
        for chunk, embedding in zip(chunks, model.generate_embeddings(chunks))
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

    queue.extend(candidates)
    
    return list(result)


def retrieve(nodes: list[Node], query: str, top_k: int, max_depth: int, min_similarity: float) -> list[Node]:
    query_embedding = model._generate_embedding(query)
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