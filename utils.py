import pickle
import torch
import numpy as np
import random as random
import os

def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy

def get_entropy_from_probabilities(probabilities):
    assert np.isclose(probabilities.sum(), 1, rtol=1.e-3)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy

def get_probs_from_outputs(arr):
    mp = {}
    for i in arr:
        if i not in mp:
            mp[i] = 0
        mp[i] += 1
    return sorted([[i, mp[i] / len(arr)] for i in mp.keys()], key=lambda x: -x[1])

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def dump_data(data, file_path):
    """
    Serializes `data` to a file specified by `file_path` using pickle.

    Parameters:
    data (any): The data to be serialized and written to the file.
    file_path (str): The path to the file where the data should be written.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data successfully dumped to {file_path}")
    except Exception as e:
        print(f"Error occurred while dumping data to {file_path}: {e}")

def load_data(file_path):
    """
    Deserializes data from a file specified by `file_path` using pickle.

    Parameters:
    file_path (str): The path to the file from which the data should be read.

    Returns:
    any: The data deserialized from the file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Data successfully loaded from {file_path}")
        return data
    except Exception as e:
        print(f"Error occurred while loading data from {file_path}: {e}")
        return None

def print_graph_from_adj_matrix(adjacency_matrix, output_file='directed_weighted_graph.png'):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    # Create a directed graph from the adjacency matrix
    G = nx.DiGraph(np.array(adjacency_matrix))

    # Get position layout for nodes
    pos = nx.spring_layout(G)

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos)

    # Draw edges with weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)

    # Remove axis
    plt.axis('off')

    # Save the graph as an image
    plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Directed graph saved as {output_file}")
    
if __name__ == "__main__":
    adjacency_matrix = [
        [0, 2, 0, 1],
        [2, 0, 3, 0],
        [0, 3, 0, 4],
        [1, 0, 4, 0]
    ]

    print_graph_from_adj_matrix(adjacency_matrix)
    