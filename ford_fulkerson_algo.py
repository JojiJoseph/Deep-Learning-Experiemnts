from copy import deepcopy
from math import e
import numpy as np
from collections import defaultdict


class Graph:
    def __init__(self) -> None:
        self.weights = {}
        self.adjacency_list = defaultdict(list)
        self.nodes = set()
    def add_edge(self, u, v, w):
        self.weights[(u, v)] = w
        self.weights[(v, u)] = 0
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)
        self.nodes.update([u, v])

sample_graph = Graph()
sample_graph.add_edge(0, 1, 5)
sample_graph.add_edge(0, 2, 10)
sample_graph.add_edge(1, 2, 7)
sample_graph.add_edge(1, 3, 10)
sample_graph.add_edge(2, 3, 5)


def BFS(graph, source, sink):
    visited = set()
    queue = [source]
    parent = {}
    while queue:
        u = queue.pop(0)
        visited.add(u)
        for v in graph.adjacency_list[u]:
            # print(u,v, graph.weights[(u, v)])
            if v not in visited and graph.weights[(u, v)] > 0:
                parent[v] = u
                queue.append(v)
    # print(visited)
    if sink in visited:
        node = sink
        flow = np.inf
        path = []
        while node != source:
            next_node = parent[node]
            flow = min(flow, graph.weights[(next_node, node)])
            path.append([next_node, node])
            node = next_node
    else:
        return 0, None
    # print(graph.weights)
    return flow, path

def ford_fulkerson(graph, source, sink):
    # resiude_graph = deepcopy(graph)
    # for edge in list(resiude_graph.weights.keys()):
    #     resiude_graph.weights[edge] = 0
    max_flow = 0
    while True:
        delta, path = BFS(graph, source, sink)
        max_flow += delta
        if not path:
            return max_flow
        for u, v in path:
            graph.weights[(u, v)] -= delta
            graph.weights[(v, u)] += delta
        # print(graph.weights)
            # resiude_graph.weights[(w, u)] += delta

print(ford_fulkerson(sample_graph, 0, 3))

graph = Graph()
graph.add_edge(0, 1, 10)
graph.add_edge(0, 2, 5)
graph.add_edge(1, 2, 15)
graph.add_edge(1, 3, 10)
graph.add_edge(2, 3, 10)
print(ford_fulkerson(graph, 0, 3))

graph = Graph()
graph.add_edge(0, 1, 10)
graph.add_edge(0, 2, 5)
graph.add_edge(1, 2, 15)
graph.add_edge(1, 3, 10)
graph.add_edge(2, 4, 10)
graph.add_edge(3, 2, 10)
graph.add_edge(3, 4, 10)

print(ford_fulkerson(graph, 0, 4))
graph = Graph()
graph.add_edge(0, 1, 16)
graph.add_edge(0, 2, 13)
graph.add_edge(1, 2, 10)
graph.add_edge(1, 3, 12)
graph.add_edge(2, 1, 4)
graph.add_edge(2, 4, 14)
graph.add_edge(3, 2, 9)
graph.add_edge(3, 5, 20)
graph.add_edge(4, 3, 7)
graph.add_edge(4, 5, 4)
print(ford_fulkerson(graph, 0, 5))

graph = Graph()
graph.add_edge((0, 'A'), (1, 'A'), 10)
graph.add_edge((0, 'A'), (1, 'B'), 5)
graph.add_edge((1, 'A'), (2, 'A'), 15)
graph.add_edge((1, 'B'), (2, 'B'), 10)
graph.add_edge((1, 'A'), (2, 'B'), 5)
graph.add_edge((2, 'A'), (3, 'B'), 10)
graph.add_edge((2, 'B'), (3, 'B'), 10)
print(ford_fulkerson(graph, (0, 'A'), (3, 'B')))

graph = Graph()
graph.add_edge((0, 'A'), (1, 'A'), 10)
graph.add_edge((0, 'A'), (1, 'B'), 9)
graph.add_edge((1, 'A'), (2, 'A'), 5)
graph.add_edge((1, 'B'), (2, 'B'), 10)
graph.add_edge((1, 'A'), (2, 'B'), 5)
graph.add_edge((2, 'A'), (4, 'B'), 5)
graph.add_edge((2, 'B'), (4, 'B'), 14)
print(ford_fulkerson(graph, (0, 'A'), (4, 'B')))


graph = Graph()
graph.add_edge((0, 'A'), (1, 'A'), 1000)
graph.add_edge((0, 'A'), (1, 'B'), 1000)
graph.add_edge((1, 'A'), (2, 'B'), 729)
graph.add_edge((1, 'B'), (2, 'B'), 1000)
print(ford_fulkerson(graph, (0, 'A'), (2, 'B')))