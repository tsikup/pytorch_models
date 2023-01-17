import dgl
import networkx as nx


def numpy_to_graph(A, type_graph="dgl", node_features=None):
    """
    Convert numpy arrays to graph

    Parameters
    ----------
    A : mxm array
        Adjacency matrix
    type_graph : str
        'dgl' or 'nx'
    node_features : dict
        Optional, dictionary with key=feature name, value=list of size m
        Allows user to specify node features

    Returns
    -------
    Graph of 'type_graph' specification
    """

    G = nx.from_numpy_array(A)

    if node_features is not None:
        for n in G.nodes():
            for k, v in node_features.items():
                G.nodes[n][k] = v[n]

    if type_graph == "nx":
        return G

    G = G.to_directed()

    if node_features is not None:
        node_attrs = list(node_features.keys())
    else:
        node_attrs = []

    g = dgl.from_networkx(G, node_attrs=node_attrs, edge_attrs=["weight"])
    return g
