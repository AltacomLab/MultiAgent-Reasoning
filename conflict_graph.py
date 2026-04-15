import numpy as np
import networkx as nx


def build_conflict_graph(predictions, threshold=0.15):
    """
    predictions: list of arrays
    """
    n = len(predictions)
    G = nx.Graph()

    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):
            diff = np.mean(np.abs(predictions[i] - predictions[j]))
            if diff > threshold:
                G.add_edge(i, j, weight=diff)

    return G


def conflict_score(G):
    return G.number_of_edges()