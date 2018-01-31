#!/bin/python3
"""This program solves Feedvack Vertex Set on Tournament Graphs"""

import sys
import itertools
import networkx as nx

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    base = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(base, r) for r in range(len(base)+1))

def loadgraph(filename):
    """Creates an networkx graph from the contents of a textfile.
     The file should be in format <node1> <node2> for every edge. The first line is skipped."""
    graph = nx.DiGraph()
    with open(filename, "r") as file:
        print(file.readline()) # skip header
        lines = file.readlines()
    for line in lines:
        node_l, node_r = line.split()
        graph.add_edge(int(node_l), int(node_r))
    print("Read {} nodes, {} edges".format(len(graph.nodes), len(graph.edges)))
    return graph

def main():
    """Main method"""
    if len(sys.argv) != 2:
        print("Usage: blatt6_feedback_vertex_set <path/to/graph.file>")
        exit(1)
    graph = loadgraph(sys.argv[1])
    node_count = len(graph.nodes)
    for k in range(0, node_count + 1):
        print("Calculating FVS for k = {}".format(k), end="\r")
        fvs = feedback_vertex_set(graph, k)
        if fvs is not False:
            print("Found a feedback vertex set with k = {}".format(k))
            exit(0)

def feedback_vertex_set(graph, k):
    """Calculates a feedback vertex set on an networkx DiGraph Tournament Graph"""
    v_i = list()
    l_feedback_vertex_set = list()
    for node in graph:
        v_i.append(node)
        g_i = graph.subgraph(v_i)
        l_feedback_vertex_set.append(node)
        if len(l_feedback_vertex_set) > k:
            l_feedback_vertex_set = feedback_vertex_set_compression(l_feedback_vertex_set, g_i)
            if l_feedback_vertex_set is False:
                return False
    return l_feedback_vertex_set

def feedback_vertex_set_compression(l_feedback_vertex_set, graph): # FVS Z, G(V, E)
    """Compresses a feedback vertex set of size k+1 to size less than k.
    returns False if it cant be compressed"""
    for fixed_nodes in powerset(l_feedback_vertex_set): # guess an X \subset Z
        complement_x = [i for i in graph if not i in fixed_nodes] # V - X
        excluded_nodes = [i for i in l_feedback_vertex_set if not i in fixed_nodes] # Y
        rest_graph = graph.subgraph(complement_x) # G - X
        fvs = disjoint_feedback_vertex_set(excluded_nodes, rest_graph) # disjoint(Y, G - X)
        if fvs is not False and len(fvs) <= (len(l_feedback_vertex_set) - len(fixed_nodes)): # |FVS| <= k - |X|
            return list(set().union(fvs, fixed_nodes))
    return False

def disjoint_feedback_vertex_set(fvs, graph):
    """Calculates a feedback vertex set of size |fvs| - 1 that is disjoint with fvs
        returns False if there is none"""
    # check if Y is DAG
    fvs_graph = graph.subgraph(fvs) # G[Y]
    if not nx.is_directed_acyclic_graph(fvs_graph): # G[Y] is not DAG?
        return False
    rest_nodes = [i for i in graph if i not in fvs] # X
    rest_graph = graph.subgraph(rest_nodes) # G[X]
    top_sort_rest = list(nx.topological_sort(rest_graph)) # G[X] sorted by order

    # test which nodes from X keep the graph acyclic
    acylic_positive = list()
    for node in rest_nodes:
        probe_nodes = list(rest_nodes)
        probe_nodes.append(node)
        probe_graph = graph.subgraph(probe_nodes)
        if nx.is_directed_acyclic_graph(probe_graph):
            acylic_positive.append(node)

    # get an ordering on X through G[Y]
    def key(node):
        if node in top_sort_rest:
            return top_sort_rest.index(node)
        return node

    y_ordering = graph.copy()
    y_ordering = y_ordering.subgraph(acylic_positive + fvs)
    for u in rest_nodes:
        for v in rest_nodes:
            if y_ordering.has_edge(u, v):
                y_ordering.remove_edge(u, v)
            if y_ordering.has_edge(v, u):
                y_ordering.remove_edge(v, u)
    
    top_sort_y_ordering = list(nx.lexicographical_topological_sort(y_ordering, key))
    
    lcs = longest_common_subsequence(top_sort_y_ordering, top_sort_rest) # F
    new_fvs = [i for i in rest_nodes if i not in lcs] # X \ F
    if len(new_fvs) < len(fvs): # |Z| < k+1
        return new_fvs
    return False

def longest_common_subsequence(seq_a, seq_b):
    """(1,2,4,5,6), (0,2,3,5,6,7) -> 2,5,6"""

    #    a1 a2 a3 a4  ...
    # b1 0  1   1  0  ...
    # b2 1
    # b3 0
    # ...

    if not seq_a or not seq_b:
        return list()

    matrix = list()
    for i in range(0, len(seq_b)):
        matrix.append(list())
        if seq_b[i] == seq_a[0]:
            matrix[i].append([seq_b[i]])
        else:
            matrix[i].append([])
        for j in range(1, len(seq_a)):
            matrix[i].append([])
    for j in range(0, len(seq_a)):
        if seq_b[0] == seq_a[j]:
            matrix[0][j] = [seq_a[j]]
        else:
            if j > 0 and len(matrix[0][j-1]) > 0:
                matrix[0][j] = list(matrix[0][j-1])
            else:
                matrix[0][j] = []

    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[i])):
            if seq_a[j] == seq_b[i]:
                matrix[i][j] = list(matrix[i-1][j-1])
                matrix[i][j].append(seq_a[j])
            else:
                if len(matrix[i][j-1]) > len(matrix[i-1][j]):
                    matrix[i][j] = list(matrix[i][j-1])
                else:
                    matrix[i][j] = list(matrix[i-1][j])

    return matrix[len(seq_b) -1][len(seq_a) - 1]

main()
