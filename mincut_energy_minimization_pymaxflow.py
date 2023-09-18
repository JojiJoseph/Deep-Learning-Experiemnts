from collections import defaultdict
from copy import deepcopy
from math import inf
import time
import cv2
import numpy as np
from traitlets import default
import maxflow


img = cv2.imread('cat.jpg', 0)
# img = cv2.resize(img, (256, 256))


# cv2.imshow("cat", img)
# cv2.waitKey(0)

mask = (img.astype(float) - 128) < 0

# cv2.imshow("mask", mask.astype(np.uint8) * 255)
# cv2.waitKey(0)

graph = defaultdict(set)
img_float = img.astype(float)

class Node:
    def __init__(self, key) -> None:
        self.connected_to_source = False
        self.connected_to_sink = False
        self.keys = set([key])
node_cache = {}
def get_node(key):
    if key in node_cache:
        return node_cache[key]
    else:
        node_cache[key] = Node(key)
        return node_cache[key]

get_node('s').connected_to_source = True
get_node('t').connected_to_sink = True

weights = defaultdict(float)

g = maxflow.Graph[float]()
# structure = np.array()
structure = []
for di in range(-3, 4):
        row = []
        for dj in range(-3, 4):
            if di == 0 and dj == 0:
                row.append(0)
            else:
                row.append(2/(np.linalg.norm([di, dj])+1))
        structure.append(row)

structure = np.array(structure)

# print(weights.shape, structure.shape)

start_time = time.time()
nodeids = g.add_grid_nodes(img.shape)
g.add_grid_edges(nodeids, weights=10, structure=structure)
g.add_grid_tedges(nodeids, img_float, 255-img_float)
g.maxflow()
sgm = g.get_grid_segments(nodeids)
mask2 = np.logical_not(sgm).astype(np.uint8) * 255
end_time = time.time()
cv2.imshow("mask2", mask2)
cv2.waitKey()
print("Time taken", end_time - start_time)
exit()


import networkx as nx

g = nx.DiGraph()
for i in range(256):
    for j in range(256):
        # graph[get_node('s')].append((get_node((i,j)), 255-img_float[i,j]))
        # graph[get_node('t')].append((get_node((i,j)), img_float[i,j]))
        # graph[get_node((i,j))].append((get_node('t'),img_float[i,j]))
        # graph[get_node((i,j))].append((get_node('s'),255-img_float[i,j]))
        graph[get_node('s')].add(get_node((i,j)))
        graph[get_node((i,j))].add(get_node('t'))
        graph[get_node((i,j))].add(get_node('s'))
        graph[get_node('t')].add(get_node((i,j)))
        weights[get_node('s'), get_node((i,j))] = 2550*(128>img_float[i,j])#/255.
        weights[get_node((i,j)), get_node('s')] = 2550*(128>img_float[i,j])#/255.
        weights[get_node('t'), get_node((i,j))] = 2550*(128<=img_float[i,j])#/255.
        weights[get_node((i,j)), get_node('t')] = 2550*(128<=img_float[i,j])#/255.
        g.add_edge(get_node('s'), get_node((i,j)), capacity=100*(255-img_float[i,j])/255.)
        g.add_edge(get_node((i,j)), get_node('t'), capacity=100*(img_float[i,j])/255.)
        # graph[get_node((i,j))].append(get_node('t'))
        for di in range(-4, 4):
            for dj in range(-4, 4):

                if 0 <= i + di < 256 and 0 <= j + dj < 256 and (di,dj) != (0,0):
                    pass
                    # graph[get_node((i,j))].add(get_node((i+di,j+dj)))#, 1))
                    # weights[get_node((i,j)), get_node((i+di,j+dj))] = 255/(np.linalg.norm([di, dj])+1)
                    # weights[get_node((i+di,j+dj)), get_node((i,j))] = 255/(np.linalg.norm([di, dj])+1)
                    # graph[get_node((i,j))].append((get_node('s'),255-img_float[i,j]))

                    g.add_edge(get_node((i,j)), get_node((i+di,j+dj)), capacity=1/(np.linalg.norm([di, dj])+1))
                    # g.add_edge(get_node((i,j)), get_node((i+di,j+dj)), capacity=255/(np.linalg.norm([di, dj])+1))

smallest_cut = inf
best_partition = None

# class Node:
    
mask = np.zeros((256,256,3))


# cv2.destroyAllWindows()

cut_value, partition = nx.minimum_cut(g, get_node('s'), get_node('t'))
for node in list(partition[1]):
    if node == get_node('s') or node == get_node('t'):
        continue
    for ij in node.keys:
        if ij == 's':
            continue
        # print(ij)
        i, j = ij
        mask[i, j] = [0,0,255]

for node in list(partition[0]):
    if node == get_node('s') or node == get_node('t'):
        continue
    for ij in node.keys:
        if ij == 's':
            continue
        # print(ij)
        i, j = ij
        mask[i, j] += [255,255,0]

cv2.imshow("mask", mask)
# cv2.imshow("mask2", mask2)
cv2.waitKey()

# for _ in range(100):
#     (graph_t, weights_t) = deepcopy((graph, weights))
#     # weights_t = deepcopy(weights)
#     # current_weight  = 0
#     while len(graph_t) > 2:
#         # select a random node
#         node = list(graph_t.keys())[np.random.randint(0, len(graph_t))]
#         # select a random edge
#         if len(graph_t[node]) == 0:
#             del graph_t[node]
#             continue
#         edge_idx = np.random.randint(0, len(graph_t[node]))
#         target_node = list(graph_t[node])[edge_idx]
#         # target_node = edge[0]
#         if target_node.connected_to_source and node.connected_to_sink:
#             continue
#         if target_node.connected_to_sink and node.connected_to_source:
#             continue
#         if node == target_node:
#             continue
#         if (node, target_node) in weights_t:
#             del weights_t[(node, target_node)]
#         if (target_node, node) in weights_t:
#             del weights_t[(target_node, node)]
#         if (node, node) in weights_t:
#             del weights_t[(node, node)]
#         if (target_node, target_node) in weights_t:
#             del weights_t[(target_node, target_node)]
#         node.keys.update(target_node.keys)
#         graph_t[node].update(graph_t[target_node])
#         if node in graph_t[node]:
#             graph_t[node].remove(node)
#         # graph_tt = deepcopy(graph_t)
#         for temp_node in list(graph_t[target_node]):
#             graph_t[temp_node].add(node)
#         # graph_t = graph_tt
#         if target_node in graph_t[node]:
#             graph_t[node].remove(target_node)
#         # graph_tt = deepcopy(graph_t)
#         for temp_node in graph_t:
#             if target_node in graph_t[temp_node]:
#                 graph_t[temp_node].remove(target_node)
#         # graph_t = graph_tt
#         # node.keys.update(target_node.keys)
#         # for edge in graph_t[target_node]:
#         #     if edge[0] != node:
#         #         graph_t[node].append(edge)

#         if target_node.connected_to_sink:
#             node.connected_to_sink = True
#         if target_node.connected_to_source:
#             node.connected_to_source = True
#         if target_node != node:
#             del graph_t[target_node]
#         print(len(graph_t))
#     # for edge in graph_t[list(graph_t.keys())[0]]:
#     #     # if edge[0] == node:
#     #     #     graph_t[node].remove(edge)
#     #     current_weight += edge[1]
#     # if current_weight < smallest_cut:
#     #     smallest_cut = current_weight
#     #     best_partition = graph_t
#     current_weight = 0
#     for weight in weights_t.values():
#         current_weight += weight
#     if current_weight < smallest_cut:
#         smallest_cut = current_weight
#         best_partition = graph_t

# mask2 = np.zeros((64,64,3))

# for node in best_partition:
#     print(node.connected_to_sink, node.connected_to_source)
#     if node.connected_to_source:
#         for ij in node.keys:
#             if ij == 's':
#                 continue
#             # print(ij)
#             i, j = ij
#             mask[i, j] = [0,0,255]
#     if node.connected_to_sink:
#         for ij in node.keys:
#             if ij == 't':
#                 continue
#             # print(ij)
#             i, j = ij
#             mask2[i, j] = [255,255,0]
# cv2.imshow("mask", mask)
# cv2.imshow("mask2", mask2)
# cv2.imshow("mask3", mask + mask2)
# cv2.waitKey()

# for node in best_partition:
#     print(smallest_cut, node.connected_to_sink, node.connected_to_source)
#     if node.connected_to_source:
#         for ij in node.keys:
#             if ij == 's':
#                 continue
#             # print(ij)
#             i, j = ij
#             mask[i, j] = [0,0,255]
#     if node.connected_to_sink:
#         for ij in node.keys:
#             if ij == 't':
#                 continue
#             # print(ij)
#             i, j = ij
#             mask2[i, j] = [255,255,0]
