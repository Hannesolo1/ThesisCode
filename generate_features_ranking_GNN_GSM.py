import random
import networkx as nx
import numpy as np
import os
import random
import pickle
import datetime
import rdflib

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

import util
from util import writeToReport, list_to_str

###################################################
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


def sample_subgraph(G, sample_size=10000):
    """Sample a subgraph from G ensuring that all sampled nodes are connected."""
    connected_nodes = [node for node, degree in G.degree() if degree > 0]
    sampled_nodes = random.sample(connected_nodes, min(sample_size, len(connected_nodes)))
    subgraph = G.subgraph(sampled_nodes)
    return subgraph


def generate_features_ranking(dataset_name,
                              data_dir='./datasets',
                              synthetic_dir='Synthetic/',
                              node_threshold=100000,
                              sample_size=50000):
    start = datetime.datetime.now()

    if 'Erdos' in dataset_name or 'Barabasi' in dataset_name:
        # Handle synthetic datasets
        graph_name = dataset_name
        G = nx.Graph()
        print(f"Processing synthetic graph: {graph_name}")

        # Load synthetic dataset
        data = torch_geometric.utils.from_networkx(
            pickle.load(open(f'{synthetic_dir}{graph_name}', 'rb'))
        )
        G = to_networkx(data, to_undirected=True)
        nodes_count = nx.number_of_nodes(G)
        edges_count = nx.number_of_edges(G)

    else:
        # Handle real datasets
        graph_name = dataset_name
        print(f"Processing real dataset: {dataset_name}")

        try:
            if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
                dataset = Planetoid(root=data_dir, name=dataset_name)
            elif dataset_name in ['Photo', 'Computers']:
                dataset = Amazon(root=data_dir, name=dataset_name)
            elif dataset_name in ['CS', 'Physics']:
                dataset = Coauthor(root=data_dir, name=dataset_name)
            elif dataset_name == 'NELL':
                dataset = NELL(root=data_dir)
            elif dataset_name in ['Cora_ML', 'DBLP']:
                dataset = CitationFull(root=data_dir, name=dataset_name)
            elif dataset_name == 'Reddit':
                dataset = Reddit(root=data_dir)
            elif dataset_name == 'Reddit2':
                dataset = Reddit2(root=data_dir)
            elif dataset_name == 'Yelp':
                dataset = Yelp(root=data_dir)
            elif dataset_name == 'AmazonProducts':
                dataset = AmazonProducts(root=data_dir)
            elif dataset_name in ['AIFB', 'AM', 'MUTAG', 'BGS']:
                dataset = Entities(root=data_dir, name=dataset_name)
            elif dataset_name in ['Actor']:
                dataset = Actor(root=data_dir)
            elif dataset_name == 'GitHub':
                dataset = GitHub(root=data_dir)
            elif dataset_name in ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions']:
                dataset = HeterophilousGraphDataset(root=data_dir, name=dataset_name)
            elif dataset_name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']:
                dataset = Twitch(root=data_dir, name=dataset_name)
            elif dataset_name in ['USA', 'Brazil', 'Europe']:
                dataset = Airports(root=data_dir, name=dataset_name)
            else:
                print(f"Dataset {dataset_name} not recognized. Skipping...")
                return
            data = dataset[0]  # Get the first data item
            G = to_networkx(data, to_undirected=True)
            nodes_count = nx.number_of_nodes(G)
            edges_count = nx.number_of_edges(G)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            return

        data = dataset[0]
        G = to_networkx(data, to_undirected=True)
        nodes_count = nx.number_of_nodes(G)
        edges_count = nx.number_of_edges(G)

        # Optionally sample if the graph is too large
        if nodes_count > node_threshold:
            print(f"Graph is too large with {nodes_count} nodes")
            return

    report_file_features = f'reports/features_synthetic/{graph_name}.csv'
    writeToReport(report_file_features,
        ('degree, degree cent., max neighbor degree, min neighbor degree, '
         'avg neighbor degree, std neighbor degree, eigenvector cent., closeness cent., '
         'harmonic cent., betweenness cent., coloring largest first, coloring smallest last, '
         'coloring independent set, coloring random sequential, coloring connected sequential dfs, '
         'coloring connected sequential bfs, edges within egonet, node clique number, number of cliques, '
         'clustering coef., square clustering coef., page rank, hubs value, triangles, core number, random '))

    start_total = datetime.datetime.now()
    metrics_count = 26
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    importance_array_gnn = np.zeros([metrics_count])
    iter_counter = 0

    # generate random walks
    walks = nx.generate_random_paths(G, iterations, walk_length)
    walks = list(walks)

    for walk in walks:
        try:
            subgraph_nodes = set()
            for n in walk:
                h = nx.ego_graph(G, n, radius=1)
                subgraph_nodes.update(h.nodes)

            subgraph_nodes = list(subgraph_nodes)[:50]
            H = G.subgraph(subgraph_nodes).copy()
            mapping = {old_label: new_label for new_label, old_label in enumerate(H.nodes())}
            H = nx.relabel_nodes(H, mapping)
            nodes_count_H = nx.number_of_nodes(H)
            edges_count_H = nx.number_of_edges(H)
            print('nodes count:', nodes_count_H)
            print('edges count:', edges_count_H)
            iter_counter += 1
        except ValueError as e:
            print(f"Warning: Skipping walk generation due to error: {e}")
            continue

        # Compute features for the random subgraph
        start_time_features = datetime.datetime.now()

        # 1) Centralities
        degree_centrality     = nx.degree_centrality(H)
        eigenvector_centrality = nx.eigenvector_centrality(H, max_iter=100, tol=1e-03)
        closeness_centrality  = nx.closeness_centrality(H)
        harmonic_centrality   = nx.harmonic_centrality(H)
        betweenness_centrality = nx.betweenness_centrality(H)

        # 2) Colorings
        coloring_largest_first = nx.coloring.greedy_color(H, strategy='largest_first')
        try:
            coloring_smallest_last = nx.coloring.greedy_color(H, strategy='smallest_last')
        except KeyError as e:
            print(f"KeyError encountered while coloring node {e}. Skipping this subgraph.")
            continue
        coloring_independent_set = nx.coloring.greedy_color(H, strategy='independent_set')
        coloring_random_sequential = nx.coloring.greedy_color(H, strategy='random_sequential')
        coloring_connected_sequential_dfs = nx.coloring.greedy_color(H, strategy='connected_sequential_dfs')
        coloring_connected_sequential_bfs = nx.coloring.greedy_color(H, strategy='connected_sequential_bfs')

        # 3) Clique & clustering
        node_clique_number = nx.node_clique_number(H)
        number_of_cliques  = {n: sum(1 for c in nx.find_cliques(H) if n in c) for n in H}
        clustering_coefficient = nx.clustering(H)
        square_clustering = nx.square_clustering(H)

        # 4) Average neighbor degree
        average_neighbor_degree = nx.average_neighbor_degree(H)

        # 5) HITS & PageRank
        hubs, authorities = nx.hits(H)
        page_rank = nx.pagerank(H)

        # 6) Core number
        core_number = nx.core_number(H)

        end_time_features = datetime.datetime.now()
        total_time_features = (end_time_features - start_time_features)
        print('total time features:', total_time_features)

        # Build X, Y
        X = np.zeros([nodes_count_H, metrics_count])
        Y = np.zeros([nodes_count_H])  # will store random labels

        ### CHANGES: Generate random labels but map them later
        comp_t_start = datetime.datetime.now()
        # Let's pick a random classes_count each time
        classes_count = np.random.randint(2, 6)  # [2..5]
        comp_t_end = datetime.datetime.now() - comp_t_start
        # Fill feature matrix and initial labels
        for i, v in enumerate(H):
            X[i][0]  = H.degree(v)
            X[i][1]  = degree_centrality[i]

            neighborhood_degrees = [H.degree(nbr) for nbr in H.neighbors(v)]
            if len(neighborhood_degrees) == 0:
                max_neighbor_degree = 0
                min_neighbor_degree = 0
                std_neighbor_degree = 0
            else:
                max_neighbor_degree = np.max(neighborhood_degrees)
                min_neighbor_degree = np.min(neighborhood_degrees)
                std_neighbor_degree = np.std(neighborhood_degrees)

            X[i][2]  = max_neighbor_degree
            X[i][3]  = min_neighbor_degree
            X[i][4]  = average_neighbor_degree[i]
            X[i][5]  = std_neighbor_degree
            X[i][6]  = eigenvector_centrality[i]
            X[i][7]  = closeness_centrality[i]
            X[i][8]  = harmonic_centrality[i]
            X[i][9]  = betweenness_centrality[i]
            X[i][10] = coloring_largest_first[i]
            X[i][11] = coloring_smallest_last[i]
            X[i][12] = coloring_independent_set[i]
            X[i][13] = coloring_random_sequential[i]
            X[i][14] = coloring_connected_sequential_dfs[i]
            X[i][15] = coloring_connected_sequential_bfs[i]

            egonet = nx.ego_graph(H, v, radius=1)
            edges_within_egonet = nx.number_of_edges(egonet)
            X[i][16] = edges_within_egonet

            X[i][17] = node_clique_number[i]
            X[i][18] = number_of_cliques[i]
            X[i][19] = clustering_coefficient[i]
            X[i][20] = square_clustering[i]
            X[i][21] = page_rank[i]
            X[i][22] = hubs[i]
            X[i][23] = nx.triangles(H, v)
            X[i][24] = core_number[i]
            X[i][25] = np.random.normal(0, 1, 1)[0]
            comp_t_start = datetime.datetime.now()
            # ### CHANGES: Generate a random label in [0..classes_count-1]
            Y[i] = np.random.randint(0, classes_count)
            comp_t_end = (datetime.datetime.now() - comp_t_start) + comp_t_end

        # Convert to int
        Y = Y.astype(int)
        comp_t_start = datetime.datetime.now()
        ### CHANGES: If subgraph has no edges or only 1 node, it can be problematic in GCN
        if H.number_of_edges() < 1 or nodes_count_H < 2:
            print("Skipping subgraph: either no edges or only 1 node.")
            continue

        # Remap labels to be continuous [0..(num_classes-1)]
        Y_unique = np.unique(Y)

        if len(Y_unique) < 2:
            # If only one label is present, skip (can't do multi-class nll_loss)
            print("Skipping subgraph: <2 unique labels.")
            continue

        # Create map from old_label -> new_label
        label_map = { old_label: new_label for new_label, old_label in enumerate(Y_unique) }
        Y_mapped = np.array([label_map[y_val] for y_val in Y])
        Y = Y_mapped
        num_classes = len(Y_unique)  # e.g. if Y_unique was [0,2,3], num_classes=3

        # Convert to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float)
        Y_tensor = torch.tensor(Y, dtype=torch.long)

        # Build edge_index from H
        edge_index = torch.tensor(list(H.edges()), dtype=torch.long).t().contiguous()

        # Create a Data object
        data_subgraph = Data(x=X_tensor, edge_index=edge_index, y=Y_tensor)

        # GCN definition
        gnn_model = GCN(num_features=metrics_count, hidden_dim=16, num_classes=num_classes)

        # Optimizer
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_subgraph = data_subgraph.to(device)
        gnn_model = gnn_model.to(device)

        data_subgraph.x.requires_grad = True

        # Train GCN
        epochs =50
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
        #GSM for feature importance link:https://arxiv.org/pdf/1905.13686
        feature_importance_gnn = data_subgraph.x.grad.abs().mean(dim=0).cpu().numpy()
        importance_array_gnn += feature_importance_gnn

        comp_t_end = (datetime.datetime.now() - comp_t_start) + comp_t_end
        comp_t_total.append(comp_t_end)
        # Write out the raw features for debugging
        for row in X:
            writeToReport(report_file_features, list_to_str(row))
        writeToReport(report_file_features, '\n')

    # Average importance across subgraphs
    if iter_counter > 0:
        importance_array_norm = importance_array_gnn / iter_counter
    else:
        importance_array_norm = np.zeros([metrics_count])

    arr_ordered = np.argsort(importance_array_norm)[::-1]
    print('Ranking (GNN-based):', arr_ordered)

    # Write out ranking
    report_file = 'data/train_GNN_GSM/ranking_synthetic.csv'
    ranking_str = ','.join(str(m) for m in arr_ordered)
    writeToReport(report_file, f"{graph_name},{nodes_count},{edges_count},{ranking_str}")

    # Write out raw importance
    report_file = 'data/train_GNN_GSM/importance_synthetic.csv'
    metrics_str = ','.join(str(np.round(m, 4)) for m in importance_array_norm)
    writeToReport(report_file, f"{graph_name},{nodes_count},{edges_count},{metrics_str}")

    # (Optional) computing_time.csv
    report_file = 'reports/computing_time.csv'
    writeToReport(report_file, 'graph_name , nodes_count , edges_count, ... (times) ...')

    end = datetime.datetime.now()
    total_time = (end - start)

    script_name = os.path.basename(__file__)  # e.g. "my_script.py"
    base_name = os.path.splitext(script_name)[0]  # e.g. "my_script"

    report_file_time = f'time/{base_name}_quick.csv'
    # Optionally write a header if the file doesn't exist. For simplicity, we'll just append.
    if not os.path.exists(report_file_time):
        writeToReport(report_file_time, "file_name,time_total")

    row_str = (
        f"{graph_name},"
        f"{total_time.total_seconds():.4f}"  # total time for dataset in seconds
    )
    writeToReport(report_file_time, row_str)

    print('Computing features time total:', total_time)

comp_t_total = []
# include datasets to generate feature ranking
datasets = []
with open('data/train/synthetic.train', 'r', encoding='utf-8') as f:
    for line in f:
        row = line.strip().split(',')
        datasets.append(row[0])

# main
iterations = 5
walk_length = 10
# datasets = [f for f in os.listdir('Synthetic/')]

for dataset in datasets:
    generate_features_ranking(dataset)


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
    f"{base_name}_quick,"
    f"{average_time_seconds:.4f},"
    f"{total_time_seconds:.4f}"  # total time for dataset in seconds
)
writeToReport(report_file_time, row_str)
