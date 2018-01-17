import os, sys, copy

class Vertex:
    def __init__(self, vid, neighbours_list):
        self.vid=vid
        self.neighbours_list=neighbours_list
        
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
        if(throw):
            raise Exception("Could not remove neighbour {} for {} from list {}".format(neighbour, self, self.neighbours_list))
    
    def __str__(self):
        return "({} {}) ".format(self.vid, self.get_degree())
        
    def __repr__(self):
        return "({} {}) ".format(self.vid, self.get_degree())
        
    def __hash__(self):
        return hash(self.vid)
        
class Graph:
    def __init__(self, n, edges, copy=False):
        self.vertices = [Vertex(i+1, list()) for i in range(n)]
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
        neighbours = set(list(vertex_a.neighbour_list)).union(set(list(vertex_b.neighbour_list)))
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
        index = cope.index(vertex)
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
    
def get_list_intersection(list1, list2):
    return_list = list()
    for e in list1:
        if e in list2:
            return_list.append(e)
    return return_list
    
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
        
def apply_rule_one(graph, vc, vertex):
    neighbour = vertex.neighbours_list[0]
    vc.append(neighbour.vid)
    graph.remove_vertex(neighbour)
    return vertex_cover(graph, vc)
    
def apply_rule_two(graph, vc, vertex):
    neighbour_a = vertex.neighbours_list[0]
    neighbour_b = vertex.neighbours_list[1]
    
    if neighbour_b in neighbour_a.neighbours_list:
        vc.append(neighbour_a.vid)
        vc.append(neighbour_b.vid)
        graph.remove_vertex(neighbour_a)
        graph.remove_vertex(neighbour_b)
        return vertex_cover(graph, vc)
    else:
        if neighbour_a.get_degree() == 2 and neighbour_b.get_degree() == 2:
            vc.append(neighbour_a.neighbours_list[0].vid)
            vc.append(neighbour_a.neighbours_list[1].vid)
            graph.remove_vertex(neighbour_a.neighbours_list[1])
            graph.remove_vertex(neighbour_a.neighbours_list[0])
            return vertex_cover(graph, vc)
        else:
            vc1 = vc[:]
            graph1 = graph.copy()
            neighbour_a1 = graph1.get_vertex(neighbour_a.vid)
            neighbour_b1 = graph1.get_vertex(neighbour_b.vid)
            vc2 = vc[:]
            graph2 = graph.copy()
            neighbour_a2 = graph2.get_vertex(neighbour_a.vid)
            neighbour_b2 = graph2.get_vertex(neighbour_b.vid)
            vertex2 = graph2.get_vertex(vertex.vid)
            
            vc1.append(neighbour_a1.vid)
            vc1.append(neighbour_b1.vid)
            graph1.remove_vertex(neighbour_a1)
            graph1.remove_vertex(neighbour_b1)
            vc1 = vertex_cover(graph1, vc1)
            
            reversed_neighbour_list_a = neighbour_a2.neighbours_list[:]
            reversed_neighbour_list_a.reverse()
            reversed_neighbour_list_b = neighbour_b2.neighbours_list[:]
            reversed_neighbour_list_b.reverse()
            
            for neighbour in reversed_neighbour_list_a:
                if neighbour is not vertex2:
                    vc2.append(neighbour.vid)
                    graph2.remove_vertex(neighbour)
            for neighbour in reversed_neighbour_list_b:
                vc2.append(neighbour.vid)
                graph2.remove_vertex(neighbour)
            vc2 = vertex_cover(graph2, vc2)
            
            if len(vc1) < len(vc2):
                return vc1
            else:
                return vc2
            
def apply_rule_three(graph, vc, vertex):
    neighbour_a = vertex.neighbours_list[0]
    neighbour_b = vertex.neighbours_list[1]
    neighbour_c = vertex.neighbours_list[2]
    vc1 = vc[:]
    graph1 = graph.copy()
    neighbour_a1 = graph1.get_vertex(neighbour_a.vid)
    neighbour_b1 = graph1.get_vertex(neighbour_b.vid)
    neighbour_c1 = graph1.get_vertex(neighbour_c.vid)
    vc2 = vc[:]
    graph2 = graph.copy()
    vertex2 = graph2.get_vertex(vertex.vid)
    neighbour_a2 = graph2.get_vertex(neighbour_a.vid)
    neighbour_b2 = graph2.get_vertex(neighbour_b.vid)
    neighbour_c2 = graph2.get_vertex(neighbour_c.vid)
    reversed_neighbour_list_b2 = neighbour_b2.neighbours_list[:]
    reversed_neighbour_list_b2.reverse()
    reversed_neighbour_list_c2 = neighbour_c2.neighbours_list[:]
    reversed_neighbour_list_c2.reverse()
    
    if len(get_list_intersection(neighbour_a.neighbours_list, neighbour_b.neighbours_list)) == 1 and len(get_list_intersection(neighbour_b.neighbours_list, neighbour_c.neighbours_list)) == 1 and len(get_list_intersection(neighbour_c.neighbours_list, neighbour_a.neighbours_list)) == 1:
        vc3 = vc[:]
        graph3 = graph.copy()
        vertex3 = graph3.get_vertex(vertex.vid)
        neighbour_a3 = graph3.get_vertex(neighbour_a.vid)
        reversed_neighbour_list_a3 = neighbour_a3.neighbours_list[:]
        reversed_neighbour_list_a3.reverse()
        
        vc1.append(neighbour_a1.vid)
        vc1.append(neighbour_b1.vid)
        vc1.append(neighbour_c1.vid)
        graph1.remove_vertex(neighbour_a1)
        graph1.remove_vertex(neighbour_b1)
        graph1.remove_vertex(neighbour_c1)
        vc1 = vertex_cover(graph1, vc1)
        
        vc2.append(neighbour_a.vid)
        for neighbour in reversed_neighbour_list_b2:
            if neighbour is not vertex2:
                vc2.append(neighbour.vid)
                graph2.remove_vertex(neighbour)
        for neighbour in reversed_neighbour_list_c2:
            vc2.append(neighbour.vid)
            graph2.remove_vertex(neighbour)
        vc2 = vertex_cover(graph2, vc2)
        
        for neighbour in reversed_neighbour_list_a3:
            vc3.append(neighbour.vid)
            graph3.remove_vertex(neighbour)
        vc3 = vertex_cover(graph3, vc3)
       
        return_vc = vc1
        if len(vc2) < len(return_vc):
            return_vc = vc2
        if len(vc3) < len(return_vc):
            return_vc = vc3      
        return return_vc
        
    elif neighbour_a in neighbour_b.neighbours_list:
        vc1.append(neighbour_a1.vid)
        vc1.append(neighbour_b1.vid)
        vc1.append(neighbour_c1.vid)
        graph1.remove_vertex(neighbour_a1)
        graph1.remove_vertex(neighbour_b1)
        graph1.remove_vertex(neighbour_c1)
        vc1 = vertex_cover(graph1, vc1)
        
        for neighbour in reversed_neighbour_list_c2:
            vc2.append(neighbour.vid)
            graph2.remove_vertex(neighbour)
        vc2 = vertex_cover(graph2, vc2)
        
        if len(vc1) < len(vc2):
            return vc1
        return vc2
        
    else:
        vc1.append(neighbour_a1.vid)
        vc1.append(neighbour_b1.vid)
        vc1.append(neighbour_c1.vid)
        graph1.remove_vertex(neighbour_a1)
        graph1.remove_vertex(neighbour_b1)
        graph1.remove_vertex(neighbour_c1)
        vc1 = vertex_cover(graph1, vc1)
        
        common_neighbours = get_list_intersection(neighbour_a2.neighbours_list, neighbour_b2.neighbours_list)
        if len(common_neighbours) == 1:
            common_neighbours = get_list_intersection(neighbour_a2.neighbours_list, neighbour_c2.neighbours_list)
        if len(common_neighbours) == 1:
            common_neighbours = get_list_intersection(neighbour_b2.neighbours_list, neighbour_c2.neighbours_list)
        common_neighbour = common_neighbours[0]
        if(common_neighbour == vertex2)
            common_neighbour = common_neighbours[1]
        vc2.append(vertex2.vid)
        vc2.append(common_neighbour.vid)
        graph2.remove_vertex(vertex2)
        graph2.remove_vertex(common_neighbour)
        vc2 = vertex_cover(graph2, vc2)
        
        if len(vc1) < len(vc2):
            return vc1
        return vc2
        
       
def apply_rule_four_or_five(graph, vc, vertex):
    vc1 = vc[:]
    graph1 = graph.copy()
    vertex1 = graph1.get_vertex(vertex.vid)
    neighbours = vertex1.neighbours_list[:]
    vc2 = vc[:]
    graph2 = graph.copy()
    vertex2 = graph2.get_vertex(vertex.vid)
    for neighbour in neighbours:
        vc1.append(neighbour.vid)
    for neighbour in neighbours:   
       graph1.remove_vertex(neighbour)
    vc1 = vertex_cover(graph1, vc1)
    
    vc2.append(vertex2.vid)
    graph2.remove_vertex(vertex2)
    vc2 = vertex_cover(graph2, vc2)
    
    if len(vc1) < len(vc2):
        return vc1
    else:
        return vc2
       
def vertex_cover(graph, vc=list()):
    if graph.vertices == list():
        return vc
        
    print vc
    print graph
        
    min_degree_vertex = graph.get_minimum_degree_vertex()
    max_degree_vertex = graph.get_maximum_degree_vertex()
    min_degree = min_degree_vertex.get_degree()
    max_degree = max_degree_vertex.get_degree()
       
    if min_degree == 0:
        graph.remove_vertex(min_degree_vertex)
        vc = vertex_cover(graph, vc)
    elif min_degree == 1:
        print "applying rule 1 with node{}".format(min_degree_vertex.vid)
        vc = apply_rule_one(graph, vc, min_degree_vertex)
    elif min_degree == 2:
        print "applying rule 2 with node {}".format(min_degree_vertex.vid)
        vc = apply_rule_two(graph, vc, min_degree_vertex)
    elif min_degree == 3:
        print "applying rule 3 with node {}".format(min_degree_vertex.vid)
        vc = apply_rule_three(graph, vc, min_degree_vertex)
    elif max_degree > 4:
        print "applying rule 4 with node {}".format(min_degree_vertex.vid)
        vc = apply_rule_four_or_five(graph, vc, min_degree_vertex)
    else:
        print "applying rule 5 with node {}".format(min_degree_vertex.vid)
        vc = apply_rule_four_or_five(graph, vc, min_degree_vertex)
    return vc
    
        
def main():
    if len(sys.argv) < 1:
        raise Exception("Provide a file name")
    for file_name in sys.argv[1:]:
        print("For {}".format(file_name))
        graph = read_graph(file_name)
        vc = vertex_cover(graph)
        print("Vertex Cover: {}".format(vc))
        
    
if __name__ == "__main__":
    main()