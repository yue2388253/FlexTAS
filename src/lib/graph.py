import networkx as nx


# Function to find neighbors within a given distance
def neighbors_within_distance(graph, node, distance):
    # Get all nodes within the specified distance
    path_lengths = nx.single_source_shortest_path_length(graph, source=node, cutoff=distance)
    # Filter out the node itself and return the list of neighbors
    return [n for n, d in path_lengths.items() if d <= distance and n != node]
