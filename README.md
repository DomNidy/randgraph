# randgraph

randgraph is a lightweight python module for generating random graphs, and rendering them with the networkx library.

 Install with: **`pip install randgraph`**

To build from source: 
```bash
python .\setup.py sdist bdist_wheel
```

## How to use it

```python
from randgraph import draw_graph, generate_graph

# Generates an undirected graph
graph = generate_graph(max_vertices=5, max_weight=15, max_out_degree=2)

# Renders out the graph
draw_graph(graph, k=5)

```

By default, graphs returned from `generate_graph()` are undirected. To create a directed graph, you can pass `directed=True` to `generate_graph()`

## Drawing and rendering a directed graph

```python
from randgraph import draw_graph, generate_graph

# Generates a directed graph
graph = generate_graph(max_vertices=5, directed=True)

# Renders out the graph
draw_graph(graph, k=5, directed=True)

```

### **Important**:

By default, `draw_graph()` assumes the graph you are passing is **undirected**, if it is not an undirected graph, you should pass `directed=True` to `draw_graph()`, the same way you do with the `generate_graph()` function.

## Easy access to underlying graph representation:

```python
from randgraph import generate_graph
# generate_graph returns an object containing an edge list representation, and an adjacency list representation
graph = generate_graph(max_vertices=5, min_weight=-5, max_weight=5)

# access the underlying graph representations
edge_list = graph.edge_list
adjacency_list = graph.adjacency_list
```
