# This Python file uses the following encoding: utf-8
import os, sys


class Vertex:

    def __init__(self, vid, neighbours_list):
        self.vid = vid
        self.neighbours_list = neighbours_list

    def get_degree(self):
        return len(self.neighbours_list)

    def add_neighbour(self, neighbour):
        neighbours = set(self.neighbours_list)
        neighbours.add(neighbour)
        self.neighbours_list = list(neighbours)

    def remove_neighbour(self, neighbour, throw=True):
        for index, self_vertex in enumerate(self.neighbours_list):
            if self_vertex.vid == neighbour.vid:
                del self.neighbours_list[index]
                return
        if throw:
            raise Exception("Could not remove neighbour {} for {} from list {}".format(neighbour, self, self.neighbours_list))

    def __str__(self):
        return "({} {}) ".format(self.vid, self.get_degree())

    def __repr__(self):
        return "({} {}) ".format(self.vid, self.get_degree())

    def __hash__(self):
        return hash(self.vid)


class Graph:

    def __init__(self, n, edges, copy=False):
        self.vertices = [Vertex(i + 1, list()) for i in range(n)]

        real_edges = list()
        for e in edges:
            if not copy:
                index_1 = e[0]
                index_2 = e[1]
            else:
                index_1 = e[0].vid
                index_2 = e[1].vid

            v1 = self.get_vertex(index_1)
            v2 = self.get_vertex(index_2)
            v1.add_neighbour(v2)
            v2.add_neighbour(v1)
            real_edges.append((v1, v2))

        self.edges = real_edges

    def get_minimum_degree_vertex(self, vertex_id=None):
        return get_minimum_degree_vertex(self.vertices, vertex_id=vertex_id)

    def get_maximum_degree_vertex(self, vertex_id=None):
        return get_maximum_degree_vertex(self.vertices, vertex_id=vertex_id)

    def get_vertex(self, vid):
        for vertex in self.vertices:
            if vertex.vid == vid:
                return vertex
        raise Exception("No vertex for id {} in {}".format(vid, self.vertices))

    def remove_vertex(self, vertex, throw=True):
        self.remove_edges(vertex, throw)
        remove_vertex_from_list(self.vertices, vertex)

    def get_edges_to_remove(self, vertex):
        edges_to_remove = list()
        for index, edge in enumerate(self.edges):
            if vertex.vid == edge[0].vid or vertex.vid == edge[1].vid:
                edges_to_remove.append(index)
        return edges_to_remove

    def add_edge(self, edge_pair):
        # if edge_pair not in self.edges:
        self.edges.append(edge_pair)
        edge_pair[0].add_neighbour(edge_pair[1])
        edge_pair[1].add_neighbour(edge_pair[0])

    def remove_edges(self, vertex, throw=True):
        remove_neighbours(vertex, throw)
        edges_to_remove = self.get_edges_to_remove(vertex)
        edges_to_remove.sort(reverse=True)
        for index in edges_to_remove:
            del self.edges[index]

    def add_vertex(self, id_, neighbours):
        vertex = Vertex(id_, neighbours)
        for neighbour in neighbours:
            neighbour.add_neighbour(vertex)
        self.vertices.append(vertex)

    def contract_vertices(self, vertex_a, vertex_b):
        neighbours = set(list(vertex_a.neighbours_list)).union(set(list(vertex_b.neighbours_list)))
        self.remove_vertex(vertex_a)
        self.remove_vertex(vertex_b, throw=False)
        neighbours = list(neighbours)
        remove_vertex_from_list(neighbours, vertex_a)
        remove_vertex_from_list(neighbours, vertex_b)
        self.add_vertex(vertex_a.vid, neighbours)

    def __str__(self):
        return "{}, {}".format(len(self.vertices), len(self.edges))

    def copy(self):
        return Graph(len(self.vertices), list(self.edges), copy=True)


def get_degree(vertex):
    return vertex.get_degree()


def remove_vertex_from_list(vertices, vertex):
    for index, self_vertex in enumerate(vertices):
        if self_vertex.vid == vertex.vid:
            del vertices[index]
            return
    raise Exception("Unknown vertex {} in {}".format(vertex, vertices))


def get_minimum_degree_vertex(vertices, vertex_id=None):
    copy = vertices
    if vertex_id is not None:
        copy = list(vertices)
        vertex = get_vertex_for_id(vertices, vertex_id)
        index = copy.index(vertex)
        del copy[index]
    return min(copy, key=get_degree)


def get_maximum_degree_vertex(vertices, vertex_id=None):
    copy = vertices
    if vertex_id is not None:
        copy = list(vertices)
        vertex = get_vertex_for_id(vertices, vertex_id)
        index = copy.index(vertex)
        del copy[index]
    return max(copy, key=get_degree)


def get_vertex_for_id(vertices, vertex_id):
    for vertex in vertices:
        if vertex.vid == vertex_id:
            return vertex
    raise Exception("Vertex {} not found in {}".format(vertex_id, vertices))


def get_edges_to_remove(h, vertex):
    index = 0
    edges_to_remove = list()
    for first, second in h.edges:
        if vertex.vid == first.vid or vertex.vid == second.vid:
            edges_to_remove.append(index)
        index = index + 1
    return edges_to_remove


def remove_neighbours(vertex, throw=True):
    for neighbour in vertex.neighbours_list:
        neighbour.remove_neighbour(vertex, throw)
    vertex.neighbours_list = list()


def remove_edges(h, vertex, throw=True):
    remove_neighbours(vertex, throw)
    edges_to_remove = get_edges_to_remove(h, vertex)
    edges_to_remove.sort(reverse=True)
    for index in edges_to_remove:
        del h.edges[index]


def get_least_similar_vertex(neighbours):
    neighbours_set = set(neighbours)
    minimum_common_neighbours = sys.maxsize
    vertex_index = -1
    for index, vertex in enumerate(neighbours):
        v_neighbours = set(vertex.neighbours_list)
        intersection_size = len(neighbours_set.intersection(v_neighbours))
        minimum_common_neighbours = min(minimum_common_neighbours, intersection_size)
        vertex_index = index
    if vertex_index == -1:
        raise Exception("Uncorrect index: {}".format(vertex_index))
    return neighbours[vertex_index]


def select_neighbour(neighbours, strategy):
    if len(neighbours) == 0:
        raise Exception("No Neighbours")
    if strategy == "min-d":
        return get_minimum_degree_vertex(neighbours)
    elif strategy == "max-d":
        return get_maximum_degree_vertex(neighbours)
    elif strategy == "least-c":
        return get_least_similar_vertex(neighbours)
    else:
        raise Exception("Unknown strategy [{}]".format(strategy))


def read_graph(file_name):
    with open(file_name, 'rb') as f:
        edges = list()
        n = 0
        for line in f:
            if n == 0:
                n, m = str(line).split(' ')
                continue
            else:
                x, y = str(line).split(' ')
                e = (int(x), int(y))
                edges.append(e)
        return Graph(int(n), edges=edges)


"""
Lower Bound
"""


def mmd(graph):
    """
    Algorithm 1 MMD(Graph G)
    1: H = G
    2: maxmin = 0
    3: while H has at least two vertices do
    4:      Select a vertex v from H that has minimum degree in H.
    5:      maxmin = max(maxmin,dH (v)).
    6:      Remove v and its incident edges from H.
    7: end while
    8: return maxmin
    """
    h = graph.copy()
    maxmin = 0
    while len(h.vertices) >= 2:
        v = h.get_minimum_degree_vertex()
        maxmin = max(maxmin, v.get_degree())
        h.remove_vertex(v)
    return maxmin


def d2D(graph):
    """
    Algorithm 2 Delta2D(Graph G)
    1: d2D = 0
    2: for all vertices v ∈ V do
    3:      H = G
    4:      while H has at least two vertices do
    5:          Select a vertex w != v from H that has minimum degree in H amongst all vertex except v.
    6:          d2D = max(d2D,dH (v)).
    7:          Remove v and its incident edges from H.
    8:      end while
    9: end for
    10: return d2D
    """
    d2d = 0
    for vertex in graph.vertices:
        h = graph.copy()
        while len(h.vertices) >= 2:
            w = h.get_minimum_degree_vertex(vertex.vid)
            d2d = max(d2d, w.get_degree())
            h.remove_vertex(w)
    return d2d


def mmd_plus(graph, strategy="least-c"):
    """
    Algorithm 3 MMD+(Graph G)
    1: H = G
    2: maxmin = 0
    3: while H has at least two vertices do
    4:      Select a vertex v from H that has minimum degree in H.
    5:      maxmin = max(maxmin,dH (v)).
    6:      Select a neighbour w of v. {A specific strategy can be used here.}
    7:      Contract the edge {v, w} in H.
    8: end while
    9: return maxmin

    Strategies to use:
    • min-d:
        select a neighbour of minimum degree. The strategy is motivated by the wish to increase the degree of small
        degree vertices as fast as possible.
    • max-d:
        select a neighbour of maximum degree. In this way, we create some neighbours of very large degree.
    • least-c:
        select the neighbour w of v such that v and w have a minimum of common neighbours. For each common neighbour x
        of v and w, contracting {v,w} let the two edges {v,x} and {w,x} become the same edge. The strategy hence tries
        to contract an edge such that as few as possible edges disappear from the graph.
    """
    h = graph.copy()
    maxmin = 0
    while len(h.vertices) >= 2:
        v = h.get_minimum_degree_vertex()
        maxmin = max(maxmin, v.get_degree())
        w = select_neighbour(v.neighbours_list, strategy)
        h.contract_vertices(v, w)
    return maxmin


def lbn(algorithm, graph):
    """
    Algorithm 4 LBN(X) (Graph G)
    1: low = X(G)
    2: repeat
    3:      changed = false
    4:      H = the (low + 1)-neighbour improved graph of G
    5:      if X(H) > low then
    6:          low = low + 1
    7:          changed = true
    8:      end if
    9: until not(changed)
    10: return low

    (k + 1)-neighbour improved graph:
        Consider the following procedure, given a graph G = (V , E) and an integer k. While there are non-adjacent
        vertices v and w that have at least k+1 common neighbours in G, add the edge {v,w} to G. Repeat this step till
        no such pair exists.
    """
    h = graph.copy()
    low = algorithm(h)
    while True:
        changed = False
        h = get_k1_improved_graph(h, low)
        if algorithm(h) > low:
            low = low + 1
            changed = True
        if not changed:
            break
    return low


def get_k1_improved_graph(graph, k):
    h = graph.copy()
    for v in graph.vertices:
        for neighbour_w in v.neighbours_list:
            for neighbour_x in v.neighbours_list:
                if neighbour_w != neighbour_x and neighbour_x not in neighbour_w.neighbours_list \
                        and len(set(neighbour_x.neighbours_list).intersection(neighbour_w.neighbours_list)) > k + 1:
                        if neighbour_x.vid > neighbour_w.vid:
                                neighbour_x, neighbour_w = neighbour_w, neighbour_x
                        h.add_edge((neighbour_x, neighbour_w))
    return h


def lbn_plus(algorithm, graph):
    # TODO
    """
    Algorithm 5 LBN+(X) (Graph G)
    1: low = X(G)
    2: repeat
    3:      changed = false
    4:      H = the (low + 1)-neighbour improved graph of G
    5:      while (X(H) low) and H has at least one edge do
    6:          Take a vertex v of minimum degree in H
    7:          Take a neighbour u of v according to least-c strategy
    8:          Contract the edge {u, v} in H
    9:          H = the (low + 1)-neighbour improved graph of H
    10:     end while
    11:     if X(H) > low then
    12:         low = low + 1
    13:         changed = true
    14:         end if
    15: until not(changed)
    16: return low
    """
    h = graph.copy()
    low = algorithm(h)
    while True:
        changed = False
        h = get_k1_improved_graph(h, low)    # TODO create (low + 1)-neighbour improved graph of G
        while algorithm(h) <= low and len(h.edges) >= 1:
            v = h.get_minimum_degree_vertex()
            u = select_neighbour(v.neighbours_list, strategy="least-c")
            h.contract_vertices(u, v)
            h = get_k1_improved_graph(h, low)  # TODO create (low + 1)-neighbour improved graph of G
        if algorithm(h) > low:
            low = low + 1
            changed = True
        if not changed:
            break
    return low


"""
Upper Bound
"""


def fill(graph, elimination_ordering):
    """
    Algorithm 1 Fill(Graph G, Elimination Ordering π)
    1:  H = G;
    2:  for i = 1 to n do
    3:      Let v = π^(−1)(i) be the ith vertex in ordering π.
    4:      for each pair of neighbours w, x of v in H with w != x, π(w) > π(v), π(x) > π(v) do
    5:          if w and x not adjacent in H then
    6:              add {w, x} to H.
    7:          end if
    8:      end for
    9:  end for
    10: return H

    A graph H = (VH, EH) is a triangulation of a graph G = (VG, EG), if H is a chordal graph that is obtained by adding
    zero or more edges to G (VG = VH, EG ⊆ EH). A triangulation H = (V, EH) is a minimal triangulation of G = (V, EG) if
    there is no triangulation of G that is a proper subgraph of H, i.e., if there is no set of edges F such that (V, F)
    is a triangulation of G with F ⊆ EH, F != EH.
    The notions for chordal graphs can now be translated to equivalent notions for treewidth.
    We first give a mechanism that adds edges to a graph to make it chordal, using an elimination ordering. Consider
    Algorithm 1. Fill(G,π) yields a graph H. One can easily observe that π is a perfect elimination ordering of H; in
    fact, we added the minimum set of edges to G such that π is a perfect elimination ordering of G. Call H the filled
    graph of G with respect to elimination ordering π, and denote this graph as Gπ+. As the filled graph H = Gπ+ has a
    perfect elimination ordering, it is a triangulation of G.

    Let G = (V , E) be a graph, and let k ≤ n be a non-negative integer. The following are equivalent.

    (i) G has treewidth at most k.
    (ii) G has a triangulation H such that the maximum size of a clique in H is at most k + 1.
    (iii) There is an elimination ordering π , such that the maximum size of a clique of Gπ+ is at most k + 1.
    """
    h = graph.copy()
    for i in range(len(graph.vertices)):
        v = h.get_vertex(elimination_ordering[i])
        for neighbour_w in v.neighbours_list:
            for neighbour_x in v.neighbours_list:
                if (neighbour_w != neighbour_x and
                        elimination_ordering.index(neighbour_w.vid) > elimination_ordering.index(v.vid) and
                        elimination_ordering.index(neighbour_x.vid) > elimination_ordering.index(v.vid)):
                    if neighbour_x not in neighbour_w.neighbours_list:
                        if neighbour_x.vid > neighbour_w.vid:
                                neighbour_x, neighbour_w = neighbour_w, neighbour_x
                        h.add_edge((neighbour_x, neighbour_w))
    return h


def permutation_to_tree_decomposition(graph, vertex_list):
    # TODO
    """
    Algorithm2 PermutationToTreeDecomposition(GraphG =(V,E), VertexList(v1,v2,...,vn))
    1:  if n = 1 then
    2:      Return a tree decomposition with one bag Xv1 = {v1}.
    3:  end if
    4:  Compute the graph G′ = (V′, E′) obtained from G by eliminating v1.
    5:  Call PermutationToTreeDecomposition(G′, (v2,v3,...,vn)) recursively, and let ({Xw | w ∈V′},T′ =(V′,F′)) be the
        returned tree decomposition.
    6:  Let vj be the lowest numbered neighbour of v1, i.e., j = min{i | {v1, vi} ∈ E}.
    7:  Construct a bag Xv1 = NG[v1]. ′
    8:  return ({Xv |v ∈V},T =(V,F))withF =F ∪{v1,vj}.
    """
    pass


def greedy_x(graph):
    # TODO
    """
    Algorithm 3 GreedyX(Graph G = (V , E))
    1:  H = G;
    2:  for i = 1 to n do
    3:      Choose a vertex v from H according to criterion X.
    4:      Let v be the ith vertex in ordering π.
    5:      Let H be the graph, obtained by eliminating v from H (make the neighbourhood of v a clique and then remove
            v.)
    6:  end for
    7:  return ordering π.

    Criterion:
        Algorithm                           Selection of next vertex
        GreedyDegree (MinimumDegree)        v = arg min_u δ_H(u)
        GreedyFillIn                        v = arg min_u φ_H(u)
        GreedyDegree+FillIn                 v = arg min_u δ_H(u) + φ_H(u)
        GreedySparsestSubgraph              v = arg min_u φ_H(u) − δ_H(u)
        GreedyFillInDegree                  v = arg min_u δ_H(u) + (1/n^2) * φ_H(u)
        GreedyDegreeFillIn                  v = arg min_u φ_H(u) + (1/n) δ_H (u)

    GreedyDegree heuristic. This heuristic was designed by Markowitz in 1957 in the context of sparse matrix
    computations [60], and is in use by many linear algebra software packages. Many studies to speed up this heuristic
    have been made; as a starting point for reading, consult e.g., [61].
    Slightly slower than GreedyDegree, but with on average slightly better bounds for the treewidth in practice is the
    GreedyFillIn heuristic (see Section 6 and [39]). In this case, we choose a vertex that causes the smallest number of
    fill edges, i.e., a vertex that has the smallest number of pairs of non-adjacent neighbours.
    GreedyDegree is motivated by the fact that we create a bag of size the degree of the chosen vertex plus one,
    GreedyFillIn by a wish not to create many new edges, as these may cause other vertices to have high degree when
    eliminated.
    GreedyDegree and GreedyFillIn are very simple heuristics, that appear to perform very well for many instances
    obtained from existing applications. For our computational evaluation in Section 6, we propose in Table 1 a few
    alternative greedy approaches that we have considered (here φH(v) denotes the number of fill edges by elimination v
    in H whereas δH(v) denotes the degree of v in H).
    Recently, new criteria have been proposed and investigated for the selection of vertices. The new treewidth
    heuristics thus obtained give in some cases improvements upon the existing heuristics.
    One such criterion was proposed by Clautiaux et al. [62,63]. Here, we compute for each vertex v first a lower bound
    on the treewidth of the graph obtained from H by eliminating v. The vertex is chosen which has the smallest value
    for the sum of twice this lower bound plus the degree in H.
    Inspired by results on preprocessing graphs for treewidth computations, Bachoore and Bodlaender [64] investigated
    several other selection criteria. To describe these, we need a few new notions.
    """
    pass


def build_tree_decomposition(graph, vertex_set):
    # TODO
    """
    Algorithm 4 BuildTreeDecomposition(Graph G = (V , E), VertexSet W )
    1:  if W = V then
    2:      return A tree decomposition with one bag containing all vertices
    3:  end if
    4:  (t, S, A, B) = FindBalancedPartition(G = (V , E), W )
    5:  if t ≡ false then
    6:      return Reject: treewidth is larger than k
    7:  end if
    8:  if (A = ∅ or B = ∅)and S ⊆ W then
    9:      Add a vertex from V \W to S
    10: end if
    11: Run BuildTreeDecomposition(G[S ∪ A], S ∪ (W ∩ A))
    12: Run BuildTreeDecomposition(G[S ∪ B], S ∪ (W ∩ B))
    13  if at least one of these runs rejects then
    14:     return Reject: treewidth is larger than k
    15: end if
    16: Take the disjoint union of the two recursively obtained tree decompositions
    17: Add a new bag x containing the vertices in S ∪ W
    18  Make x adjacent to the root nodes of the two recursively obtained tree decompositions return The just computed
        tree decomposition of G with x as root.
    """


def refine_tree_decomposition(graph, tree_decomposition):
    # TODO
    """
    Algorithm 5 RefineTreeDecomposition(Graph G = (V, E), TreeDecomposition ({Xi, i ∈ I}, T = (I, F)))
    1:  while ∃i ∈ I such that |Xi| maximal and G[Xi] does not induce a clique do
    2:      ConstructgraphHi withvertexsetXi andedgeset{{v,w}∈Xi ×Xi|{v,w}∈E∨∃j̸=i:v,w∈Xj}
    3:      Compute minimum separator S ⊂ Xi in Hi; let W1, . . . , Wr define the r connected components of Hi[Xi \ S]
    4:      SetI′ =I\{i}∪{i0,...,ir}
    5:      SetX′=Xjforallj̸=i,X′ =S,X′ =Wq∪Sforq=1,...,r j i0iq
    6:      Set F′ =F \{{i,j}|j ∈NT(i)}∪{{i0,iq}|q =1,...,r}∪{{j,iqj}|j ∈NT(i)} where qj ∈{1,...,r} such that Xi ∩Xj ⊆
    7:      Wqj ∪S
    8:  end while
    9:  return Tree decomposition ({Xj′, j ∈ I′}, T′ = (I′, F′)).
    """
    pass


def minimal_triangulation(graph):
    # TODO
    """
    Algorithm 6 MinimalTriangulation(Graph G = (V , E))
    1:  G′ = G;
    2:  while G′ is not a chordal graph do
    3:      Choose a minimal separator S in G′ that is not a clique
    4:      Let G′ be the graph, obtained by completing S in G′
    5:  end while
    6:  return G′.
    """
    pass


"""
Main
"""


def lower_bound(graph):
    print("MMD lower bound: {}".format(mmd(graph)))
    print("d2D lower bound: {}".format(d2D(graph)))
    print("MMD+ lower bound with min-d: {}".format(mmd_plus(graph, "min-d")))
    print("MMD+ lower bound with max-d: {}".format(mmd_plus(graph, "max-d")))
    print("MMD+ lower bound with least-c: {}".format(mmd_plus(graph, "least-c")))
    print("LBN lower bound for {}: {}".format(mmd.__name__, lbn(mmd, graph)))
    print("LBN lower bound for {}: {}".format(d2D.__name__, lbn(d2D, graph)))
    print("LBN lower bound for {}: {}".format(mmd_plus.__name__, lbn(mmd_plus, graph)))
    # print("LBN+ lower bound for {}: {}".format(mmd.__name__, lbn_plus(mmd, graph)))
    # print("LBN+ lower bound for {}: {}".format(d2D.__name__, lbn_plus(d2D, graph)))
    # print("LBN+ lower bound for {}: {}".format(mmd_plus.__name__, lbn_plus(mmd_plus, graph)))


def upper_bound(graph):
    elimination_ordering = [i + 1 for i in range(len(graph.vertices))]
    h = fill(graph, elimination_ordering)
    vertex = h.get_maximum_degree_vertex()
    print("Fill upper bound with {}: {}".format(elimination_ordering, len(vertex.neighbours_list) - 1))

    # permutation_to_tree_decomposition(graph, vertex_list)
    # greedy_x(graph)
    # build_tree_decomposition(graph, vertex_set)
    # refine_tree_decomposition(graph, tree_composition)
    # minimal_triangulation(graph)


def main():
    if len(sys.argv) < 1:
        raise Exception("Provide a file name")
    for file_name in sys.argv[1:]:
        print("For {}".format(file_name))
        graph = read_graph(file_name)
        lower_bound(graph)
        upper_bound(graph)


if __name__ == "__main__":
    main()
