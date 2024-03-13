import random as rand
import networkx as nx
import dataclasses
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple


@dataclasses.dataclass
class Edge:
    """
    Representation of an edge in a graph as an object. Contains the vertex the edge is pointing to and the weight of the edge.
    """

    vertex: int  # the vertex the edge is pointing to
    weight: int | None  # the weight of the edge

    def __hash__(self) -> int:
        return hash((self.vertex, self.weight))

    def __getitem__(self, index):
        return (self.vertex, self.weight)[index]


@dataclasses.dataclass
class CreateEdgesResult:
    """
    The return value of the generate_graph function. Contains an adjacency list and an edge list representation of the graph.
    """

    adjacency_list: Dict[int, List[Tuple[int, int]] | List[Edge]]
    edge_list: List[List[Tuple[int, int]] | List[Edge]]


def generate_graph(
    max_vertices=5,
    directed=False,
    max_out_degree=2,
    min_weight: float = 1,
    max_weight: float = 15,
    allow_floating_point_weights=False,
) -> CreateEdgesResult:
    """
    Creates random graph, returns an adjacency list and an edge list representation of the graph.

    max_vertices: the maximum number of vertices in the graph
    directed: whether the graph is directed or not (true for directed, false for undirected)
    max_out_degree: the maximum out degree of a vertex in the graph

    min_weight: the minimum weight of an edge in the graph
    max_weight: the maximum weight of an edge in the graph
    allow_floating_point_weights: whether the weights of the edges can be floating point numbers or not
    """
    assert max_vertices > 2, "max_vertices must be greater than 1"
    assert min_weight < max_weight, "min_weight must be less than max_weight"

    # probably should make this more readable but it looks funny
    edges = [
        list(
            set(
                [
                    (
                        Edge(
                            Utils.rand_int_except(0, max_vertices - 1, exclude=i),
                            round(
                                rand.random() * (max_weight - min_weight) + min_weight,
                                0 if not allow_floating_point_weights else 1,
                            ),
                        )
                    )
                    for j in range(rand.randint(1, max_out_degree))
                ]
            )
        )
        for i in range(max_vertices)
    ]

    # for undirected graphs, add the reverse edges
    if not directed:
        for i in range(len(edges)):
            for edge in edges[i]:
                # if the vertex the edge is pointing to does not already have an edge to the current vertex, add one
                # if it does, replace the weight of the edge with the current edge's weight
                if not any(e.vertex == i for e in edges[edge.vertex]):
                    edges[edge.vertex].append(Edge(i, edge.weight))

                # replace the weight of the reverse edge with the current edge's weight
                else:
                    for e in edges[edge.vertex]:
                        if e.vertex == i:
                            e.weight = edge.weight

                edges[edge.vertex].append(Edge(i, edge.weight))

    # when a node u has two different edges to a node v, remove the edges after the first one
    # ensures we dont have duplicate edges
    for i in range(len(edges)):
        # store the vertices that have been visited from this vertex
        curr_edges = []

        # edges to remove after we finish iterating through the edges
        edges_to_remove = []
        for j in range(len(edges[i])):
            if edges[i][j].vertex not in curr_edges:
                curr_edges.append(edges[i][j].vertex)
            else:
                # if we find an edge to a vertex that has already been visited, mark it for removal
                edges_to_remove.append(edges[i][j])

        for edge_to_remove in edges_to_remove:
            edges[i].remove(edge_to_remove)

    adjacency_list = {i: edges[i] for i in range(len(edges))}

    return CreateEdgesResult(adjacency_list, edges)


# Infer type of adjacency_list
def draw_graph(
    graph: (
        CreateEdgesResult
        | Dict[int, List[Tuple[int, int]] | List[Edge]]
        | List[List[Edge] | List[Tuple[int, int]]]
    ),
    directed: bool = False,
    k: float = 5,
) -> None:
    """
    Draws a graph using the networkx library.

    graph: The graph to draw, can be an adjacency list, an edge list, or a CreateEdgesResult object.
    directed: Whether the graph is directed or not (true for directed, false for undirected) (Should match the input into the generate_graph function which returned the graph passed into this function.)
    k: The optimal distance between nodes. If None the distance is set to 1/sqrt(n) where n is the number of nodes. Increase this value to move nodes farther apart.

    The input graph can be an adjacency list, an edge list, or a CreateEdgesResult object.
    Note: Each edge is a tuple of the form (vertex, weight) where vertex is the vertex the edge is pointing to.
    """

    if isinstance(graph, CreateEdgesResult):
        adj = graph.adjacency_list
    elif isinstance(graph, list):
        adj = {i: graph[i] for i in range(len(graph))}

    G = nx.DiGraph() if directed else nx.Graph()

    for node in adj:
        for edge in adj[node]:
            G.add_edge(node, edge.vertex, weight=edge.weight, with_labels=True)
            pos = nx.spring_layout(G, k=k)  # positions for all nodes

    edge_labels = nx.get_edge_attributes(G, "weight")
    # draw arrows

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos, with_labels=True)
    plt.show()


class Utils:
    @staticmethod
    def count_edges(
        graph: (
            Dict[int, List[Tuple[int, int]] | List[Edge]]
            | List[List[Edge] | List[Tuple[int, int]]]
            | CreateEdgesResult
        ) = None,
    ) -> int:
        if isinstance(graph, dict):
            return sum(len(graph[i]) for i in graph)
        elif isinstance(graph, list):
            return sum(len(graph[i]) for i in range(len(graph)))
        elif isinstance(graph, CreateEdgesResult):
            return sum(len(graph.adjacency_list[i]) for i in graph.adjacency_list)
        else:
            raise ValueError(
                "Invalid input, expected an adjacency list, edge list, or CreateEdgesResult object."
            )

    @staticmethod
    def rand_int_except(a: int, b: int, exclude: int) -> int:
        """
        Returns a random integer between a and b (inclusive) that is not equal to exclude.
        """
        num = rand.randint(a, b)
        while num == exclude:
            num = rand.randint(a, b)
        return num
