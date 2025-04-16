# generate synthetic graphs using the following parameters:
# base graph: Erdos or Barabasi, nodes_count in the base graph, edges count in Erdos, or m0 in Barabasi,
# count of motifs, motifs shape: path, house, grid, star, cycle, noisy edges.
# generated graphs are saved under Synthetic/

base_graph_type = 'Barabasi'
base_graph_nodes_count_list = [10000]
base_graph_edges_count_list = [5]
motifs_count_list = [800]
motifs_shapes_list = ['path', 'house', 'grid', 'star']
noisy_edges_count_list = [600]
generated_graphs = []
graphs_ind = 1

import random
import networkx as nx
import numpy as np
import os
import random
import pickle
import datetime
import util
from util import writeToReport, list_to_str
import torch_geometric
from torch_geometric.utils import to_networkx

def create_motif(shape):
    G = nx.Graph()
    if (shape == 'house'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 2, 'x': 1}),
                             (counter + 2, {'y': 2, 'x': 1}),
                             (counter + 3, {'y': 3, 'x': 1}), (counter + 4, {'y': 3, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter, counter + 2))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 1, counter + 3))
        G.add_edge(*(counter + 2, counter + 4))
        G.add_edge(*(counter + 3, counter + 4))
        motif_size = 5

    if (shape == 'star'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 2, 'x': 1}),
                             (counter + 2, {'y': 2, 'x': 1}),
                             (counter + 3, {'y': 2, 'x': 1}), (counter + 4, {'y': 2, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter, counter + 2))
        G.add_edge(*(counter, counter + 3))
        G.add_edge(*(counter, counter + 4))
        motif_size = 5

    if (shape == 'path'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 2, 'x': 1}),
                             (counter + 2, {'y': 2, 'x': 1}),
                             (counter + 3, {'y': 2, 'x': 1}), (counter + 4, {'y': 1, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 2, counter + 3))
        G.add_edge(*(counter + 3, counter + 4))
        motif_size = 5

    if (shape == 'cycle'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 1, 'x': 1}),
                             (counter + 2, {'y': 1, 'x': 1}),
                             (counter + 3, {'y': 1, 'x': 1}), (counter + 4, {'y': 1, 'x': 1}),
                             (counter + 5, {'y': 1, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 2, counter + 3))
        G.add_edge(*(counter + 3, counter + 4))
        G.add_edge(*(counter + 4, counter + 5))
        G.add_edge(*(counter + 5, counter))
        motif_size = 6

    if (shape == 'grid'):
        motifs_attributes = [(counter, {'y': 1, 'x': 1}), (counter + 1, {'y': 2, 'x': 1}),
                             (counter + 2, {'y': 1, 'x': 1}),
                             (counter + 3, {'y': 2, 'x': 1}), (counter + 4, {'y': 3, 'x': 1}),
                             (counter + 5, {'y': 2, 'x': 1}),
                             (counter + 6, {'y': 1, 'x': 1}), (counter + 7, {'y': 2, 'x': 1}),
                             (counter + 8, {'y': 1, 'x': 1})]
        G.add_nodes_from(motifs_attributes)
        G.add_edge(*(counter, counter + 1))
        G.add_edge(*(counter + 1, counter + 2))
        G.add_edge(*(counter + 3, counter + 4))
        G.add_edge(*(counter + 4, counter + 5))
        G.add_edge(*(counter + 6, counter + 7))
        G.add_edge(*(counter + 7, counter + 8))
        G.add_edge(*(counter, counter + 3))
        G.add_edge(*(counter + 3, counter + 6))
        G.add_edge(*(counter + 1, counter + 4))
        G.add_edge(*(counter + 4, counter + 7))
        G.add_edge(*(counter + 2, counter + 5))
        G.add_edge(*(counter + 5, counter + 8))
        motif_size = 9

    return G, motif_size


def create_base_graph(nodes_count, edges_count):
    if (base_graph_type == 'Barabasi'):
        base_graph = nx.barabasi_albert_graph(nodes_count, edges_count)
    if (base_graph_type == 'Erdos'):
        #base_graph = nx.gnp_random_graph(nodes_count, 0.1)
        #base_graph = nx.gnp_random_graph(nodes_count, 0.5)
        #base_graph = nx.gnp_random_graph(nodes_count, 0.8)
        base_graph = nx.gnp_random_graph(nodes_count, 0.05)

        base_graph = nx.gnm_random_graph(nodes_count, edges_count)
    nx.set_node_attributes(base_graph, 1, 'x')
    nx.set_node_attributes(base_graph, 0, 'y')

    return base_graph



for n in base_graph_nodes_count_list:
    for m in base_graph_edges_count_list:
        for motifs_count in motifs_count_list:
            for motifs_shape in motifs_shapes_list:
                for noisy_edges_count in noisy_edges_count_list:

                    start = datetime.datetime.now()
                    print('graph: ' + str(graphs_ind))
                    base_graph = create_base_graph(n, m)
                    counter = n
                    #attach motifs
                    for motif in range(0, motifs_count):
                        motif_graph, motif_size = create_motif(motifs_shape)
                        base_graph = nx.union(base_graph, motif_graph)
                        random_base_node = random.randint(0, n - 1)
                        random_motif_node = random.randint(counter, counter + 4)
                        base_graph.add_edge(*(random_base_node, random_motif_node))
                        counter += motif_size
                    # for loop noisy edges
                    noisy_edges_count = int(nx.number_of_edges(base_graph) * 0.01)
                    for e in range(noisy_edges_count):
                        noisy_edge_node_1 = random.randint(0, counter - 1)
                        noisy_edge_node_2 = random.randint(0, counter - 1)
                        base_graph.add_edge(*(noisy_edge_node_1, noisy_edge_node_2))

                    base_graph.remove_edges_from(nx.selfloop_edges(base_graph))
                    Gcc = sorted(nx.connected_components(base_graph), key=len, reverse=True)
                    base_graph = base_graph.subgraph(Gcc[0])

                    graph_name = base_graph_type + '_' + str(n) + '_' + str(
                        m) + '_' + motifs_shape + '_' + str(motifs_count) + '_' + str(
                        noisy_edges_count) + '_' + str(graphs_ind) + '.pickle'
                    pickle.dump(base_graph, open('Synthetic/' + graph_name, 'wb'))
                    generated_graphs.append(graph_name)

                    end = datetime.datetime.now()
                    total_time = datetime.timedelta()
                    total_time = (end - start)
                    print('Generating time: ' + str(total_time))
                    graphs_ind += 1


