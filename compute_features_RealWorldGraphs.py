import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets import LINKXDataset, HeterophilousGraphDataset, CitationFull, Coauthor
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import datetime
import os
from torch_geometric.datasets import GNNBenchmarkDataset, Airports, AttributedGraphDataset


def list_to_str(lst):
    return ','.join(map(str, lst))

def writeToReport(filename, content):
    with open(filename, 'a') as f:
        f.write(content + '\n')

def compute_features(data, graph_name):
    start = datetime.datetime.now()
    print(f"Processing dataset: {graph_name}")
    nodes_count = data.num_nodes
    edges_count = data.num_edges

    report_file_features = f'data/synthetic/{graph_name}.csv'
    os.makedirs('data/synthetic', exist_ok=True)
    writeToReport(report_file_features,
                  'degree, degree cent., max neighbor degree, min neighbor degree, avg neighbor degree, std neighbor degree, '
                  'eigenvector cent., closeness cent., harmonic cent., betweenness cent., '
                  'coloring largest first, coloring smallest last, coloring independent set, coloring random sequential,'
                  'coloring connected sequential dfs, coloring connected sequential bfs, edges within egonet,'
                  ' node clique number, number of cliques, clustering coef., square clustering coef., page rank, hubs value,'
                  ' triangles, core number, random, Target')

    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    subgraph_nodes = list(G.nodes)
    metrics_count = 26
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Initialize metrics
    degree_centrality = nx.degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100, tol=1e-03)
    closeness_centrality = nx.closeness_centrality(G)
    harmonic_centrality = nx.harmonic_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    coloring_largest_first = nx.coloring.greedy_color(G, strategy='largest_first')
    coloring_smallest_last = nx.coloring.greedy_color(G, strategy='smallest_last')
    coloring_independent_set = nx.coloring.greedy_color(G, strategy='independent_set')
    coloring_random_sequential = nx.coloring.greedy_color(G, strategy='random_sequential')
    coloring_connected_sequential_dfs = nx.coloring.greedy_color(G, strategy='connected_sequential_dfs')
    coloring_connected_sequential_bfs = nx.coloring.greedy_color(G, strategy='connected_sequential_bfs')
    node_clique_number = nx.node_clique_number(G)
    cliques = list(nx.find_cliques(G))
    number_of_cliques = {n: sum(1 for c in cliques if n in c) for n in G}
    clustering_coefficient = nx.clustering(G)
    square_clustering = nx.square_clustering(G)
    average_neighbor_degree = nx.average_neighbor_degree(G)
    hubs, _ = nx.hits(G, max_iter=100, tol=1e-03)
    page_rank = nx.pagerank(G)
    core_number = nx.core_number(G)

    X = np.zeros([nx.number_of_nodes(G), metrics_count])
    Y = np.zeros([nx.number_of_nodes(G)])
    for i, v in enumerate(G):
        X[i][0] = G.degree(v)
        X[i][1] = degree_centrality[v]
        neighborhood_degrees = [G.degree(n) for n in nx.neighbors(G, v)]
        if neighborhood_degrees:
            X[i][2] = np.max(neighborhood_degrees)
            X[i][3] = np.min(neighborhood_degrees)
            X[i][4] = average_neighbor_degree[v]
            X[i][5] = np.std(neighborhood_degrees)
        else:
            X[i][2:6] = 0
        X[i][6] = eigenvector_centrality[v]
        X[i][7] = closeness_centrality[v]
        X[i][8] = harmonic_centrality[v]
        X[i][9] = betweenness_centrality[v]
        X[i][10] = coloring_largest_first[v]
        X[i][11] = coloring_smallest_last[v]
        X[i][12] = coloring_independent_set[v]
        X[i][13] = coloring_random_sequential[v]
        X[i][14] = coloring_connected_sequential_dfs[v]
        X[i][15] = coloring_connected_sequential_bfs[v]
        egonet = nx.ego_graph(G, v, radius=1)
        X[i][16] = nx.number_of_edges(egonet)
        X[i][17] = node_clique_number[v]
        X[i][18] = number_of_cliques[v]
        X[i][19] = clustering_coefficient[v]
        X[i][20] = square_clustering[v]
        X[i][21] = page_rank[v]
        X[i][22] = hubs[v]
        X[i][23] = nx.triangles(G, v)
        X[i][24] = core_number[v]
        X[i][25] = np.random.normal(0, 1)
        Y[i] = data.y[i].item() if hasattr(data, 'y') else 0  # Ensure it's a scalar

        writeToReport(report_file_features, list_to_str(X[i]) + ',' + str(Y[i]))

    end = datetime.datetime.now()
    total_time = end - start
    print(f"Total time for {graph_name}: {total_time}")

    report_file = 'reports/computing_time.csv'
    os.makedirs('reports', exist_ok=True)
    writeToReport(report_file, f"{graph_name},{nodes_count},{edges_count},{total_time}")

# Load and process datasets
# Airports datasets: USA, Brazil, Europe
from torch_geometric.datasets import LINKXDataset

data_path = './datasets'
airports_datasets = ['Europe', 'USA']


for dataset_name in airports_datasets:
    dataset = Airports(root=data_path, name=dataset_name)
    data = dataset[0]
    compute_features(data, dataset_name)

# Attributed Graph Datasets: Actor, GitHub
# from torch_geometric.datasets import Actor, GitHub
#
#
# dataset = GitHub(root=data_path)
# data = dataset[0]
# compute_features(data, "GitHub")
#
# dataset = Actor(root=data_path)
# data = dataset[0]
# compute_features(data, "Actor")



# Heterophilous Graph Datasets: Minesweeper, Roman-empire, Amazon-ratings
from torch_geometric.datasets import HeterophilousGraphDataset

# hetero_datasets = ['minesweeper']
# #hetero_datasets = ['amazon-ratings']
#
#
# for dataset_name in hetero_datasets:
#     dataset = HeterophilousGraphDataset(root=data_path, name=dataset_name)
#     data = dataset[0]
#     compute_features(data, dataset_name)

# CitationFull dataset: DBLP
# from torch_geometric.datasets import CitationFull
#
# citation_dataset = CitationFull(root=data_path, name='DBLP')
# data = citation_dataset[0]
# compute_features(data, 'DBLP')

