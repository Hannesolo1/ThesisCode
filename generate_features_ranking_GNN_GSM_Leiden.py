import random
import networkx as nx
import numpy as np
import os
import pickle
import datetime
import torch_geometric
from torch_geometric.utils import to_networkx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import igraph as ig
import leidenalg
import util
from util import writeToReport, list_to_str
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler  # For feature scaling
import torch
import torch.nn.functional as F
from torch_geometric.datasets import (
    Planetoid, Amazon, Coauthor, Reddit, Reddit2, Yelp, AmazonProducts, Entities,
    CitationFull, NELL, Actor, GitHub, HeterophilousGraphDataset, Twitch, Airports
)
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx

# Simple 2-layer GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

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
    if 'Erdos' in graph_name:
        walk_length_local = 20  # Make the walk longer for Erdos to capture more nodes
        resolution = 0.5  # Lower resolution for Erdos
        subgraph_size = 50
        if nodes_count < 50:
            subgraph_size = nodes_count
        elif nodes_count > 15000:
            resolution = 0.1
    else:
        walk_length_local = walk_length
        resolution = 1.0  # Default resolution for Barabasi
        subgraph_size = 40

    # Generate initial random walks
    walks = nx.generate_random_paths(G, iterations, walk_length_local)
    walks = list(walks)
    print(f" - Generated {len(walks)} random walks.\n")

    def find_best_merge_target(small_comm, Y, H):
        """Find the "closest" community to merge into based on edge connectivity.
        Returns the label of the community to merge into."""
        small_nodes = [i for i, label in enumerate(Y) if label == small_comm]
        neighbor_edges_count = defaultdict(int)

        # Count edges to other communities
        for node in small_nodes:
            for neighbor in H.neighbors(node):
                neigh_comm = Y[neighbor]
                if neigh_comm != small_comm:
                    neighbor_edges_count[neigh_comm] += 1

        if neighbor_edges_count:
            # Pick the community with the highest number of connecting edges
            best_target = max(neighbor_edges_count, key=neighbor_edges_count.get)
            return best_target
        else:
            # If no edges, as a fallback, merge with the largest community
            community_counts = Counter(Y)
            largest_community = None
            largest_size = -1
            for c, size in community_counts.items():
                if c != small_comm and size > largest_size:
                    largest_size = size
                    largest_community = c
            return largest_community

    def merge_small_communities(Y, H, min_community_size):
        """Recursively or iteratively merge small communities from smallest to largest,
        always picking the closest community for merging."""
        while True:
            community_counts = Counter(Y)
            # Identify small communities
            small_communities = [(c, sz) for c, sz in community_counts.items() if sz < min_community_size]

            if not small_communities:
                # No small communities left
                break

            # Sort small communities by size ascending
            small_communities.sort(key=lambda x: x[1])

            merged_any = False
            for (small_comm, sz) in small_communities:
                # Find best target community
                target_community = find_best_merge_target(small_comm, Y, H)
                if target_community is not None and target_community != small_comm:
                    # Perform the merge
                    Y = Y.copy()
                    Y[Y == small_comm] = target_community
                    # Re-map labels to ensure continuity and no gaps
                    Y = remap_labels(Y)
                    merged_any = True
                else:
                    # If we cannot find a suitable target, we stop the merging attempt
                    pass

            if not merged_any:
                # No merges were possible in this iteration
                break

        return Y

    def remap_labels(Y):
        """Ensure labels start from 0 and are continuous."""
        unique_labels = np.unique(Y)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        Y_mapped = np.array([label_mapping[label] for label in Y])
        return Y_mapped

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

        # If too few nodes, skip early
        if nodes_count_H < 10:
            print("[WARNING] Subgraph too small, skipping.")
            # Add a random walk if possible
            if extra_walk_count < max_extra_walks:
                new_walk = next(nx.generate_random_paths(G, 1, walk_length_local))
                walks.append(new_walk)
                extra_walk_count += 1
                walk_idx += 1
                continue
            else:
                print("[INFO] Could not successfully rank 3 subgraphs after 8 extra tries. Skipping graph.")
                return

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

        comp_t_start = datetime.datetime.now()
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

        comp_t_end = datetime.datetime.now() - comp_t_start
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

        comp_t_start = datetime.datetime.now()

        # Convert to int
        Y = Y.astype(int)

        ### CHANGES: If subgraph has no edges or only 1 node, it can be problematic in GCN
        if H.number_of_edges() < 1 or nodes_count_H < 2:
            print("Skipping subgraph: either no edges or only 1 node.")
            continue

        # Convert to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float)
        Y_tensor = torch.tensor(Y, dtype=torch.long)

        # Build edge_index from H
        edge_index = torch.tensor(list(H.edges()), dtype=torch.long).t().contiguous()

        # Create a Data object
        data_subgraph = Data(x=X_tensor, edge_index=edge_index, y=Y_tensor)

        # GCN definition
        num_classes = len(set(Y))
        gnn_model = GCN(num_features=metrics_count, hidden_dim=16, num_classes=num_classes)

        # Optimizer
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_subgraph = data_subgraph.to(device)
        gnn_model = gnn_model.to(device)

        data_subgraph.x.requires_grad = True

        # Train GCN
        epochs = 200
        gnn_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = gnn_model(data_subgraph)
            loss = F.nll_loss(out, data_subgraph.y)
            loss.backward()
            optimizer.step()

        # Evaluate feature importance (gradient-based)
        gnn_model.eval()
        data_subgraph.x.requires_grad = True
        optimizer.zero_grad()
        out = gnn_model(data_subgraph)
        loss = F.nll_loss(out, data_subgraph.y)
        loss.backward()

        feature_importance_gnn = data_subgraph.x.grad.abs().mean(dim=0).cpu().numpy()
        importance_array += feature_importance_gnn
        iter_counter += 1
        successful_subgraphs += 1

        comp_t_end = datetime.datetime.now() - comp_t_start + comp_t_end

        comp_t_total.append(comp_t_end)

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
        report_file = 'data/train_GNN_GSM_Leiden/ranking_synthetic_testing.csv'
        ranking_str = ''.join(str(m) + ',' for m in arr_ordered)
        writeToReport(report_file, graph_name + ',' + str(nodes_count) + ',' + str(edges_count) + ',' + ranking_str)

        report_file = 'data/train_GNN_GSM_Leiden/importance_synthetic_testing.csv'
        metrics_str = ''.join(str(np.round(m, 4)) + ',' for m in importance_array_norm)
        writeToReport(report_file, graph_name + ',' + str(nodes_count) + ',' + str(edges_count) + ',' + metrics_str)
    else:
        print("[INFO] No valid iterations were performed for this dataset.")

    end = datetime.datetime.now()
    total_time = (end - start)

    script_name = os.path.basename(__file__)  # e.g. "my_script.py"
    base_name = os.path.splitext(script_name)[0]  # e.g. "my_script"

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


# include datasets to generate feature ranking
comp_t_total = []
datasets = []
with open('data/train/synthetic.train', 'r', encoding='utf-8') as f:
    for line in f:
        row = line.strip().split(',')
        datasets.append(row[0])


# Main
iterations = 5
walk_length = 10
# datasets = [f for f in os.listdir('Synthetic/') if f.endswith('.pickle')]
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