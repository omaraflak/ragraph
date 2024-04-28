import fire
import pickle
import prompts
import numpy as np
from dataclasses import dataclass
from model import Model, Embedding
from common import get_model, read_data_chunks, cosine_similarity


@dataclass
class Node:
    text: str
    embedding: Embedding
    neighbors: list[tuple['Node', Embedding]]


    def __hash__(self) -> int:
        return int(np.prod(self.embedding))


def compute_edges(model: Model, pairs: list[tuple[str, str]]) -> list[Embedding]:
    requests = [prompts.chunks_relationship.format(chunk1=a, chunk2=b) for a, b in pairs]
    return model.generate_embeddings(model.generate_texts(requests))


def create_nodes(model: Model, chunks: list[str]) -> list[Node]:
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

    for (a, b, _), edge_embedding in zip(edges, compute_edges(model, pairs)):
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


def retrieve(model: Model, nodes: list[Node], query: str, top_k: int, max_depth: int, min_similarity: float) -> list[Node]:
    query_embedding = model.generate_embeddings([query])[0]
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


def create_graph(input_txt_file: str = 'source.txt', output_bin_file: str = 'graph.bin', model_name: str = 'gemini'):
    model = get_model(model_name)
    chunks = read_data_chunks(input_txt_file)
    nodes = create_nodes(model, chunks)
    save_graph(nodes, output_bin_file)


def query_graph(query: str, graph_bin_file: str = 'graph.bin', model_name: str = 'gemini'):
    model = get_model(model_name)
    nodes = load_graph(graph_bin_file)
    results = retrieve(model, nodes, query, top_k=3, max_depth=4, min_similarity=0.3)
    for node in results:
        print(node.text)


if __name__ == '__main__':
    fire.Fire({
        'create': create_graph,
        'query': query_graph
    })
