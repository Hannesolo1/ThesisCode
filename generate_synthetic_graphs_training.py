
import csv
import random
import networkx as nx
import numpy as np
import os
import pickle
import datetime
import torch_geometric
from torch_geometric.utils import to_networkx

# If you have util.py that you import from:
import util
from util import writeToReport, list_to_str

###############################################################################
# 1) Create Motif
###############################################################################
def create_motif(shape, counter):
    G = nx.Graph()
    if shape == 'house':
        motifs_attributes = [
            (counter,     {'y': 1, 'x': 1}),
            (counter + 1, {'y': 2, 'x': 1}),
            (counter + 2, {'y': 2, 'x': 1}),
            (counter + 3, {'y': 3, 'x': 1}),
            (counter + 4, {'y': 3, 'x': 1})
        ]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(counter, counter + 1)
        G.add_edge(counter, counter + 2)
        G.add_edge(counter + 1, counter + 2)
        G.add_edge(counter + 1, counter + 3)
        G.add_edge(counter + 2, counter + 4)
        G.add_edge(counter + 3, counter + 4)
        motif_size = 5

    elif shape == 'star':
        motifs_attributes = [
            (counter,     {'y': 1, 'x': 1}),
            (counter + 1, {'y': 2, 'x': 1}),
            (counter + 2, {'y': 2, 'x': 1}),
            (counter + 3, {'y': 2, 'x': 1}),
            (counter + 4, {'y': 2, 'x': 1})
        ]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(counter, counter + 1)
        G.add_edge(counter, counter + 2)
        G.add_edge(counter, counter + 3)
        G.add_edge(counter, counter + 4)
        motif_size = 5

    elif shape == 'path':
        motifs_attributes = [
            (counter,     {'y': 1, 'x': 1}),
            (counter + 1, {'y': 2, 'x': 1}),
            (counter + 2, {'y': 2, 'x': 1}),
            (counter + 3, {'y': 2, 'x': 1}),
            (counter + 4, {'y': 1, 'x': 1})
        ]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(counter, counter + 1)
        G.add_edge(counter + 1, counter + 2)
        G.add_edge(counter + 2, counter + 3)
        G.add_edge(counter + 3, counter + 4)
        motif_size = 5

    elif shape == 'cycle':
        motifs_attributes = [
            (counter,     {'y': 1, 'x': 1}),
            (counter + 1, {'y': 1, 'x': 1}),
            (counter + 2, {'y': 1, 'x': 1}),
            (counter + 3, {'y': 1, 'x': 1}),
            (counter + 4, {'y': 1, 'x': 1}),
            (counter + 5, {'y': 1, 'x': 1})
        ]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(counter, counter + 1)
        G.add_edge(counter + 1, counter + 2)
        G.add_edge(counter + 2, counter + 3)
        G.add_edge(counter + 3, counter + 4)
        G.add_edge(counter + 4, counter + 5)
        G.add_edge(counter + 5, counter)
        motif_size = 6

    elif shape == 'grid':
        motifs_attributes = [
            (counter,     {'y': 1, 'x': 1}),
            (counter + 1, {'y': 2, 'x': 1}),
            (counter + 2, {'y': 1, 'x': 1}),
            (counter + 3, {'y': 2, 'x': 1}),
            (counter + 4, {'y': 3, 'x': 1}),
            (counter + 5, {'y': 2, 'x': 1}),
            (counter + 6, {'y': 1, 'x': 1}),
            (counter + 7, {'y': 2, 'x': 1}),
            (counter + 8, {'y': 1, 'x': 1})
        ]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(counter,     counter + 1)
        G.add_edge(counter + 1, counter + 2)
        G.add_edge(counter + 3, counter + 4)
        G.add_edge(counter + 4, counter + 5)
        G.add_edge(counter + 6, counter + 7)
        G.add_edge(counter + 7, counter + 8)
        G.add_edge(counter,     counter + 3)
        G.add_edge(counter + 3, counter + 6)
        G.add_edge(counter + 1, counter + 4)
        G.add_edge(counter + 4, counter + 7)
        G.add_edge(counter + 2, counter + 5)
        G.add_edge(counter + 5, counter + 8)
        motif_size = 9

    else:
        raise ValueError("Unknown motif shape: {}".format(shape))

    return G, motif_size


###############################################################################
# 2) Create Base Graph
###############################################################################
def create_base_graph(graph_type, nodes_count, edges_count):
    if graph_type == 'Barabasi':
        base_graph = nx.barabasi_albert_graph(nodes_count, edges_count)
    elif graph_type == 'Erdos':
        base_graph = nx.gnm_random_graph(nodes_count, edges_count)
    else:
        raise ValueError("Unknown base graph type: {}".format(graph_type))

    # default attributes for all nodes
    nx.set_node_attributes(base_graph, 1, 'x')
    nx.set_node_attributes(base_graph, 0, 'y')

    return base_graph


###############################################################################
# 3) Parse the filename of the form:
#    baseGraphType_nodes_edges_motifShape_motifCount_noisyEdges_graphInd.pickle
###############################################################################
def parse_filename(filename):
    """
    Example filename:
      Barabasi_700_12_house_80_440_246.pickle
      => base_graph_type = 'Barabasi'
         base_graph_nodes_count = 700
         base_graph_edges_count = 12
         motifs_shape = 'house'
         motifs_count = 80
         noisy_edges_count = 440
         graphs_ind = 246
    """
    # remove '.pickle'
    name = filename.replace('.pickle', '')
    parts = name.split('_')
    if len(parts) != 7:
        raise ValueError(
            f"Expected 7 parts in filename, got {len(parts)}: {parts}"
        )

    base_graph_type       = parts[0]
    base_graph_nodes_count = int(parts[1])
    base_graph_edges_count = int(parts[2])
    motifs_shape           = parts[3]
    motifs_count           = int(parts[4])
    noisy_edges_count      = int(parts[5])
    graphs_ind             = int(parts[6])

    return (base_graph_type,
            base_graph_nodes_count,
            base_graph_edges_count,
            motifs_shape,
            motifs_count,
            noisy_edges_count,
            graphs_ind)

###############################################################################
# 4) Main loop: Read CSV, parse each row, and generate each graph
###############################################################################
def generate_graphs_from_csv(csv_file_path, output_dir='Synthetic2'):
    """
    Reads the CSV file line by line, and for each row:
      1) Extracts the .pickle filename from the first column.
      2) Parses its structure to get all parameters.
      3) Generates the graph and saves it with the exact same filename into
         the output_dir.
    """

    # Make sure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generated_graphs = []

    with open(csv_file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty lines
            if not row:
                continue

            # The first column is the .pickle filename
            pickle_filename = row[0].strip()

            # Parse the filename to extract parameters
            (base_graph_type,
             base_graph_nodes_count,
             base_graph_edges_count,
             motifs_shape,
             motifs_count,
             noisy_edges_count,
             graphs_ind) = parse_filename(pickle_filename)

            # Now create the base graph
            start = datetime.datetime.now()
            print(f"Generating graph index: {graphs_ind} -> {pickle_filename}")

            base_graph = create_base_graph(
                base_graph_type,
                base_graph_nodes_count,
                base_graph_edges_count
            )

            # We'll attach 'motifs_count' motifs of shape 'motifs_shape'
            counter = base_graph_nodes_count
            for _ in range(motifs_count):
                motif_graph, motif_size = create_motif(motifs_shape, counter)
                # Union them
                base_graph = nx.union(base_graph, motif_graph)
                # Connect a random node in the base graph to a random node in the motif
                random_base_node  = random.randint(0, base_graph_nodes_count - 1)
                random_motif_node = random.randint(counter, counter + motif_size - 1)
                base_graph.add_edge(random_base_node, random_motif_node)
                # Advance the counter for next motif
                counter += motif_size

            # Add noisy edges
            # (If you want it to be a fraction of the current edges, you can do so.
            #  But here we directly use noisy_edges_count from the filename.)
            for _ in range(noisy_edges_count):
                noisy_edge_node_1 = random.randint(0, counter - 1)
                noisy_edge_node_2 = random.randint(0, counter - 1)
                base_graph.add_edge(noisy_edge_node_1, noisy_edge_node_2)

            # Remove self loops (if any)
            base_graph.remove_edges_from(nx.selfloop_edges(base_graph))

            # Keep only the largest connected component
            Gcc = sorted(nx.connected_components(base_graph), key=len, reverse=True)
            base_graph = base_graph.subgraph(Gcc[0]).copy()

            # Save using the same filename in output_dir
            output_path = os.path.join(output_dir, pickle_filename)
            with open(output_path, 'wb') as pf:
                pickle.dump(base_graph, pf)

            generated_graphs.append(pickle_filename)

            end = datetime.datetime.now()
            total_time = (end - start)
            print(f"  -> Generated {pickle_filename} in {total_time}.")

    return generated_graphs



generate_graphs_from_csv('data/train/training_graphs.csv', output_dir='Synthetic')

