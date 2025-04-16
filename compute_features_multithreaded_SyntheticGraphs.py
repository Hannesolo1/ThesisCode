import os
import collections
import numpy as np
import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch_geometric
from torch_geometric.utils import to_networkx
import networkx as nx
import csv
import datetime
import random
from sklearn.feature_selection import mutual_info_classif
import torch_geometric.transforms as T
from sklearn.ensemble import RandomForestClassifier
import util
from util import writeToReport
from sklearn.model_selection import cross_val_score
import pickle
import multiprocessing as mp

# Global lock variable
lock = None


def collect_datasets(graph_name_list,
                     synthetic_dir='Synthetic/',
                     data_synthetic_dir='data/synthetic/'):
    datasets = []

    # Ensure the directories exist
    if not os.path.isdir(synthetic_dir):
        raise FileNotFoundError(f"Synthetic directory '{synthetic_dir}' does not exist.")
    if not os.path.isdir(data_synthetic_dir):
        raise FileNotFoundError(f"Data synthetic directory '{data_synthetic_dir}' does not exist.")

    # Get a list of already processed filenames
    processed_files = os.listdir(data_synthetic_dir)

    for graph_name in graph_name_list:
        # Check if graph_name is NOT contained in any of the processed filenames
        if not any(graph_name in filename for filename in processed_files):
            # Find all files in 'synthetic_dir' that contain the graph_name
            matching_files = [f for f in os.listdir(synthetic_dir) if graph_name in f]
            datasets.extend(matching_files)  # Collect all matching datasets
            print(f"Graph '{graph_name}' not found in processed files. Added {len(matching_files)} matching files.")
        else:
            print(f"Graph '{graph_name}' already processed. Skipping.")
    print(f"Total datasets to process: {len(datasets)}")
    return datasets

def init(l):
    global lock
    lock = l

def list_to_str(lst):
    return ','.join(map(str, lst))

def compute_features(dataset):
    global lock
    print(f"Process ID: {os.getpid()} is processing graph: {dataset}")
    start = datetime.datetime.now()
    graph_name = dataset
    data = torch_geometric.utils.from_networkx(pickle.load(open('Synthetic2/' + dataset, 'rb')))
    nodes_count = data.num_nodes
    edges_count = data.num_edges

    report_file_features = 'data/synthetic2/' + graph_name + '.csv'
    writeToReport(report_file_features,
                  'degree, degree cent., max neighbor degree, min neighbor degree, avg neighbor degree, std neighbor degree, '
                  'eigenvector cent., closeness cent., harmonic cent., betweenness cent., '
                  'coloring largest first, coloring smallest last, coloring independent set, coloring random sequential,'
                  'coloring connected sequential dfs, coloring connected sequential bfs, edges within egonet,'
                  ' node clique number, number of cliques, clustering coef., square clustering coef., page rank, hubs value,'
                  ' triangles, core number, random, Target')

    G = to_networkx(data, to_undirected=True)
    metrics_count = 26
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    H = G

    # Compute features
    start_time_features = datetime.datetime.now()

    degree_centrality = nx.degree_centrality(H)
    eigenvector_centrality = nx.eigenvector_centrality(H, max_iter=100, tol=1e-03)
    closeness_centrality = nx.closeness_centrality(H)
    harmonic_centrality = nx.harmonic_centrality(H)
    betweenness_centrality = nx.betweenness_centrality(H)
    coloring_largest_first = nx.coloring.greedy_color(H, strategy='largest_first')
    coloring_smallest_last = nx.coloring.greedy_color(H, strategy='smallest_last')
    coloring_independent_set = nx.coloring.greedy_color(H, strategy='independent_set')
    coloring_random_sequential = nx.coloring.greedy_color(H, strategy='random_sequential')
    coloring_connected_sequential_dfs = nx.coloring.greedy_color(H, strategy='connected_sequential_dfs')
    coloring_connected_sequential_bfs = nx.coloring.greedy_color(H, strategy='connected_sequential_bfs')
    node_clique_number = nx.node_clique_number(H)
    number_of_cliques = {n: sum(1 for c in nx.find_cliques(H) if n in c) for n in H}
    clustering_coefficient = nx.clustering(H)
    square_clustering = nx.square_clustering(H)
    average_neighbor_degree = nx.average_neighbor_degree(H)
    hubs, authorities = nx.hits(H)
    page_rank = nx.pagerank(H)
    core_number = nx.core_number(H)

    end_time_features = datetime.datetime.now()
    total_time_features = end_time_features - start_time_features

    X = np.zeros([nx.number_of_nodes(H), metrics_count])
    Y = np.zeros([nx.number_of_nodes(H)])
    for i, v in enumerate(H):
        X[i][0] = H.degree(v)
        X[i][1] = degree_centrality[v]
        neighborhood_degrees = [H.degree(n) for n in nx.neighbors(H, v)]
        if len(neighborhood_degrees) == 0:
            max_neighbor_degree = 0
            min_neighbor_degree = 0
            std_neighbor_degree = 0
        else:
            max_neighbor_degree = np.max(neighborhood_degrees)
            min_neighbor_degree = np.min(neighborhood_degrees)
            std_neighbor_degree = np.std(neighborhood_degrees)
        X[i][2] = max_neighbor_degree
        X[i][3] = min_neighbor_degree
        X[i][4] = average_neighbor_degree[v]
        X[i][5] = std_neighbor_degree
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
        edges_within_egonet = nx.number_of_edges(egonet)

        X[i][16] = edges_within_egonet
        X[i][17] = node_clique_number[v]
        X[i][18] = number_of_cliques[v]
        X[i][19] = clustering_coefficient[v]
        X[i][20] = square_clustering[v]
        X[i][21] = page_rank[v]
        X[i][22] = hubs[v]
        X[i][23] = nx.triangles(H, v)
        X[i][24] = core_number[v]
        X[i][25] = np.random.normal(0, 1)
        Y[i] = data.y[v].item()  # Ensure it's a scalar

    for i, x in enumerate(X):
        writeToReport(report_file_features, list_to_str(x) + ',' + str(Y[i]))
    writeToReport(report_file_features, '\n')

    end = datetime.datetime.now()
    total_time = end - start

    # Prepare data for the report
    report_file = 'reports/computing_time.csv'
    data_line = f"{graph_name},{nodes_count},{edges_count},{total_time_features.total_seconds()}"

    with lock:
        writeToReport(report_file, data_line)

if __name__ == '__main__':
    import os

    datasets = [f for f in os.listdir('Synthetic/') if 'Barabasi' in f or 'Erdos' in f]

    data_dir = "./datasets"
    os.makedirs(data_dir, exist_ok=True)

    report_file = 'reports/computing_time.csv'
    if not os.path.exists('reports'):
        os.makedirs('reports')
    writeToReport(report_file, 'graph_name,nodes_count,edges_count,total_time_features_seconds')

    # Create a Manager and a Lock
    manager = mp.Manager()
    lock = manager.Lock()

    # Get the number of CPU cores
    num_cores = mp.cpu_count()-11
    print(f"Number of CPU cores available: {num_cores}")

    # Start multiprocessing pool with initializer
    with mp.Pool(processes=num_cores, initializer=init, initargs=(lock,)) as pool:
        pool.map(compute_features, datasets)
