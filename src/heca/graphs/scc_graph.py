import networkx as nx
import numpy as np

import networkx as nx


def get_scc_clusters(G):
    return list(nx.strongly_connected_components(G))


def build_graph(labels, matrix, threshold=0.7):
    G = nx.DiGraph()

    # add nodes
    for l in labels:
        G.add_node(l)

    # add directed edges
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if matrix[i, j] >= threshold:
                G.add_edge(labels[i], labels[j], weight=matrix[i, j])

    return G
