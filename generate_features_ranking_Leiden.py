import random
import networkx as nx
import numpy as np
import os
import pickle
import datetime
import torch_geometric
import matplotlib
matplotlib.use('Agg')  # For saving figures headlessly
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # <-- This line is needed

from torch.fx.experimental.symbolic_shapes import resolve_unbacked_bindings
from torch_geometric.utils import to_networkx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import igraph as ig
import leidenalg
import util
from util import writeToReport, list_to_str
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler  # For feature scaling

def generate_features_ranking(dataset):
    start = datetime.datetime.now()
    graph_name = dataset

    print("===================================================")
    print(f"Processing dataset: {graph_name}")
    print("===================================================")

    G = nx.Graph()
    data = torch_geometric.utils.from_networkx(pickle.load(open('Synthetic/' + graph_name, 'rb')))
    G = to_networkx(data, to_undirected=True)
    nodes_count = nx.number_of_nodes(G)
    edges_count = nx.number_of_edges(G)

    data_dir = 'reports/features_synthetic/'
    report_file_features = os.path.join(data_dir, f'{graph_name}.csv')

    writeToReport(report_file_features,
                  'degree, degree cent., max neighbor degree, min neighbor degree, avg neighbor degree, std neighbor degree, '
                  'eigenvector cent., closeness cent., harmonic cent., betweenness cent., '
                  'coloring largest first, coloring smallest last, coloring independent set, coloring random sequential,'
                  'coloring connected sequential dfs, coloring connected sequential bfs, edges within egonet,'
                  ' node clique number, number of cliques, clustering coef., square clustering coef., page rank, hubs value,'
                  ' triangles, core number')

    metrics_count = 26
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    importance_array = np.zeros(metrics_count)
    iter_counter = 0

    # Increase subgraph size for Erdos, for example
    resolution = 1.0
    subgraph_size = 50

    # Generate initial random walks
    walks = nx.generate_random_paths(G, iterations, walk_length)
    walks = list(walks)
    print(f" - Generated {len(walks)} random walks.\n")


    # Variables for the new logic
    successful_subgraphs = 0
    extra_walk_count = 0
    max_extra_walks = 20

    walk_idx = 0
    while walk_idx < len(walks) and successful_subgraphs < iterations:
        print("---------------------------------------------------")
        print(f"[INFO] Processing subgraph from walk {walk_idx+1}/{len(walks)}:")
        walk = walks[walk_idx]

        subgraph_nodes = set()
        for n in walk:
            h = nx.ego_graph(G, n, radius=1)
            subgraph_nodes.update(h.nodes)

        subgraph_nodes = list(subgraph_nodes)[:subgraph_size]
        H = G.subgraph(subgraph_nodes).copy()
        mapping = {old_label: new_label for new_label, old_label in enumerate(H.nodes())}
        H = nx.relabel_nodes(H, mapping)
        nodes_count_H = nx.number_of_nodes(H)
        edges_count_H = nx.number_of_edges(H)
        print(f" - Subgraph nodes: {nodes_count_H}")
        print(f" - Subgraph edges: {edges_count_H}")


        nodes_list = sorted(H.nodes())
        num_nodes = len(nodes_list)

        start_time_features = datetime.datetime.now()

        # Compute features
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
        # For number_of_cliques, this could be expensive. We will compute it once.
        all_cliques = list(nx.find_cliques(H))
        number_of_cliques = {n: sum(1 for c in all_cliques if n in c) for n in H}
        clustering_coefficient = nx.clustering(H)
        square_clustering = nx.square_clustering(H)
        average_neighbor_degree = nx.average_neighbor_degree(H)
        hubs, authorities = nx.hits(H)
        page_rank = nx.pagerank(H)
        core_number = nx.core_number(H)

        end_time_features = datetime.datetime.now()
        total_time_features = end_time_features - start_time_features
        print('total time features:', total_time_features)

        comb_t_start = datetime.datetime.now()
        # Perform community detection using Leiden algorithm
        H_ig = ig.Graph(n=nodes_count_H)
        H_ig.add_edges(list(H.edges()))

        partition = leidenalg.find_partition(
            H_ig,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution
        )

        # Extract community labels
        Y = np.array(partition.membership)
        community_counts = Counter(Y)
        print(" - Original community sizes:", dict(community_counts))

        comb_t_end = datetime.datetime.now() - comb_t_start

        # Compute feature matrix X
        X = np.zeros([len(Y), metrics_count])
        for idx, v in enumerate(nodes_list):
            X[idx][0] = H.degree(v)
            X[idx][1] = degree_centrality[v]
            neighborhood_degrees = [H.degree(n) for n in H.neighbors(v)]
            if len(neighborhood_degrees) == 0:
                max_neighbor_degree = 0
                min_neighbor_degree = 0
                std_neighbor_degree = 0
            else:
                max_neighbor_degree = np.max(neighborhood_degrees)
                min_neighbor_degree = np.min(neighborhood_degrees)
                std_neighbor_degree = np.std(neighborhood_degrees)
            X[idx][2] = max_neighbor_degree
            X[idx][3] = min_neighbor_degree
            X[idx][4] = average_neighbor_degree[v]
            X[idx][5] = std_neighbor_degree
            X[idx][6] = eigenvector_centrality[v]
            X[idx][7] = closeness_centrality[v]
            X[idx][8] = harmonic_centrality[v]
            X[idx][9] = betweenness_centrality[v]
            X[idx][10] = coloring_largest_first[v]
            X[idx][11] = coloring_smallest_last[v]
            X[idx][12] = coloring_independent_set[v]
            X[idx][13] = coloring_random_sequential[v]
            X[idx][14] = coloring_connected_sequential_dfs[v]
            X[idx][15] = coloring_connected_sequential_bfs[v]

            egonet = nx.ego_graph(H, v, radius=1)
            edges_within_egonet = nx.number_of_edges(egonet)

            X[idx][16] = edges_within_egonet
            X[idx][17] = node_clique_number[v]
            X[idx][18] = number_of_cliques[v]
            X[idx][19] = clustering_coefficient[v]
            X[idx][20] = square_clustering[v]
            X[idx][21] = page_rank[v]
            X[idx][22] = hubs[v]
            X[idx][23] = nx.triangles(H, v)
            X[idx][24] = core_number[v]
            X[idx][25] = np.random.normal(0, 1, 1)[0]



        # If we reach this point, we have a valid subgraph
        comb_t_start = datetime.datetime.now()
        model = RandomForestClassifier()
        model.fit(X, Y)
        arr = model.feature_importances_
        importance_array += arr
        iter_counter += 1
        successful_subgraphs += 1
        comb_t_end = (datetime.datetime.now() - comb_t_start) + comb_t_end
        comp_t_total.append(comb_t_end)
        # Optionally, write features to report
        for x in X:
            writeToReport(report_file_features, list_to_str(x))
        writeToReport(report_file_features, '\n')

        walk_idx += 1

    # After processing walks
    if successful_subgraphs < iterations:
        print("[INFO] Could not successfully rank 3 subgraphs after 8 extra tries. Skipping graph.")
        return

    if iter_counter > 0:
        importance_array_norm = importance_array / iter_counter
        arr_ordered = np.argsort(importance_array_norm)[::-1]
        print("[RESULT] Feature importance ranking (descending):", arr_ordered)
        report_file = 'data/train_Leiden/ranking_synthetic_testing.csv'
        ranking_str = ''.join(str(m) + ',' for m in arr_ordered)
        writeToReport(report_file, graph_name + ',' + str(nodes_count) + ',' + str(edges_count) + ',' + ranking_str)

        report_file = 'data/train_Leiden/importance_synthetic_testing.csv'
        metrics_str = ''.join(str(np.round(m, 4)) + ',' for m in importance_array_norm)
        writeToReport(report_file, graph_name + ',' + str(nodes_count) + ',' + str(edges_count) + ',' + metrics_str)
    else:
        print("[INFO] No valid iterations were performed for this dataset.")

    end = datetime.datetime.now()
    total_time = (end - start)

    script_name = os.path.basename(__file__)
    base_name = os.path.splitext(script_name)[0]

    report_file_time = f'time/{base_name}.csv'
    # Optionally write a header if the file doesn't exist. For simplicity, we'll just append.
    if not os.path.exists(report_file_time):
        writeToReport(report_file_time, "file_name,time_total")

    row_str = (
        f"{graph_name},"
        f"{total_time.total_seconds():.4f}"  # total time for dataset in seconds
    )
    writeToReport(report_file_time, row_str)

    print('Computing features time total:', total_time)


comp_t_total = []  # <-- Add this line to store computation times
# include datasets to generate feature ranking
datasets = []
with open('data/train/synthetic.train', 'r', encoding='utf-8') as f:
    for line in f:
        row = line.strip().split(',')
        datasets.append(row[0])

# main
iterations = 5
walk_length = 10
for dataset in datasets:
    generate_features_ranking(dataset)
print("Code runs successfully")
total_time = datetime.timedelta()  # Initialize total_time as a zero timedelta

# Sum up all the times in comp_t_total
for t in comp_t_total:
    total_time += t

# Now calculate the average in seconds
average_time_seconds = total_time.total_seconds() / len(comp_t_total) if len(comp_t_total) > 0 else 0

# Log the total time and average time
print(f"Total computation time: {total_time}")
print(f"Average computation time (seconds): {average_time_seconds}")

# Write to the report file
script_name = os.path.basename(__file__)  # Get the script name (e.g. "my_script.py")
base_name = os.path.splitext(script_name)[0]  # Get the base name (e.g. "my_script")
report_file_time = 'time/average_time.csv'

# Optionally write a header if the file doesn't exist. For simplicity, we'll just append.
if not os.path.exists(report_file_time):
    writeToReport(report_file_time, "file_name,average_time,time_total")

total_time_seconds = total_time.total_seconds()
# Write the average time to the report file
row_str = (
    f"{base_name},"
    f"{average_time_seconds:.4f},"
    f"{total_time_seconds:.4f}"  # total time for dataset in seconds
)
writeToReport(report_file_time, row_str)