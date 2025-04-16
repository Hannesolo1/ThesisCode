import random
import networkx as nx
import numpy as np
import os
import random
import pickle
import datetime
import rdflib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm, inv


from torch_geometric.datasets import (
    Planetoid, Amazon, Coauthor, Reddit, Reddit2, Yelp, AmazonProducts, Entities,
    CitationFull, NELL, Actor, GitHub, HeterophilousGraphDataset, Twitch, Airports
)
import util
from util import writeToReport, list_to_str
from sklearn.ensemble import RandomForestClassifier
import torch_geometric
from torch_geometric.utils import to_networkx

import numpy as np
from scipy.linalg import sqrtm, inv
from scipy.linalg import fractional_matrix_power


def normalize_simplex(W):
    """
    Projects W onto the probability simplex by normalizing rows.
    Ensures each row sums to 1 while preserving the structure of W.
    """
    return W / (np.sum(W, axis=1, keepdims=True) + 1e-10)


def compute_gamma_s(W, s):
    """
    Compute the self-assignment normalization matrix gamma_s(W)
    following the geodesic interpolation from the paper.

    Args:
        W (np.ndarray): The assignment matrix (n_nodes, n_clusters).
        s (float): Interpolation exponent (0 <= s <= 1).

    Returns:
        np.ndarray: The computed gamma_s(W).
    """
    # Compute the cluster normalization matrix C(W)
    C_W = np.diag(np.sum(W, axis=0))

    # Compute intermediate terms for geodesic interpolation
    C_W_inv_sqrt = inv(sqrtm(C_W + 1e-7 * np.eye(W.shape[1])))  # C(W)^(-1/2)
    WTW = W.T @ W  # W^T W
    M_s = C_W_inv_sqrt @ WTW @ C_W_inv_sqrt  # Transformed space

    # Compute M_s^s via matrix power

    M_s_s = fractional_matrix_power(M_s, s)  # Correct matrix exponentiation

    # Compute final gamma_s(W)
    gamma_s = sqrtm(C_W) @ M_s_s @ sqrtm(C_W)  # Transform back

    return gamma_s


def self_assignment_flow(K, W, max_iters=50, lr=0.1, s=0.5):
    """
    Perform the self-assignment flow dynamics following the full formulation from the paper.

    Args:
        K (np.ndarray): The similarity/kernel matrix of shape (n_nodes, n_nodes).
        W (np.ndarray): The assignment matrix of shape (n_nodes, n_clusters).
        max_iters (int): Number of iteration steps.
        lr (float): Learning rate for numerical integration.
        s (float): Exponent for computing gamma_s(W), balancing between probabilistic and subspace clustering.

    Returns:
        np.ndarray: Updated assignment matrix W (n_nodes, n_clusters).
    """
    n_nodes, n_clusters = W.shape

    for iteration in range(max_iters):
        # Compute gamma_s(W) following geodesic interpolation
        gamma_s = compute_gamma_s(W, s)

        # Compute inverse of gamma_s
        inv_gamma_s = inv(gamma_s + 1e-7 * np.eye(n_clusters))  # Small epsilon for numerical stability

        # Compute the self-assignment matrix A_s(W)
        A_s = W @ inv_gamma_s @ W.T

        # Compute the gradient of the optimization objective
        grad = 2 * K @ (W @ inv_gamma_s)

        # Perform numerical integration step using Riemannian manifold constraints
        W = W * np.exp(lr * grad)  # Exponential map update to stay on the simplex
        W = normalize_simplex(W)  # Ensure row-wise normalization

    return W


def rbf_kernel(X, gamma=1.0):
    # X is shape (n, d)
    # Return shape (n, n)
    X_sq = np.sum(X**2, axis=1, keepdims=True)  # shape (n,1)
    dists_sq = X_sq + X_sq.T - 2.0 * (X @ X.T)  # shape (n,n)
    K = np.exp(-gamma * dists_sq)
    return K

#This is to introduce a bot of noise back into the labels
def introduce_label_noise(Y, noise_fraction=0.05):
    """
    Randomly flip a fraction of labels in-place.

    Args:
        Y (np.ndarray): shape (n,), cluster labels from SAF.
        noise_fraction (float): fraction of nodes to flip label.

    Returns:
        np.ndarray: Y with random flips.
    """
    if noise_fraction <= 0 or len(np.unique(Y)) <= 1:
        return Y

    n = len(Y)
    n_flips = int(noise_fraction * n)
    # Randomly pick the indices to flip
    flip_indices = np.random.choice(n, size=n_flips, replace=False)

    # For each chosen index, assign a different label at random
    unique_labels = np.unique(Y)
    for idx in flip_indices:
        current_label = Y[idx]
        # pick a label different from current_label
        alt_labels = unique_labels[unique_labels != current_label]
        Y[idx] = np.random.choice(alt_labels)

    return Y


def generate_features_ranking(dataset_name, data_dir='./datasets', synthetic_dir='Synthetic/', node_threshold=100000,
                              sample_size=50000):
    start = datetime.datetime.now()

    if 'Erdos' in dataset_name or 'Barabasi' in dataset_name:
        # Handle synthetic datasets
        graph_name = dataset_name
        G = nx.Graph()
        print(f"Processing synthetic graph: {graph_name}")

        # Load synthetic dataset (assuming it's a pickle file in the Synthetic directory)
        data = torch_geometric.utils.from_networkx(pickle.load(open(f'{synthetic_dir}{graph_name}', 'rb')))
        G = torch_geometric.utils.to_networkx(data, to_undirected=True)
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
                return  # Skip if the dataset is not recognized
            data = dataset[0]  # Get the first data item
            G = torch_geometric.utils.to_networkx(data, to_undirected=True)
            nodes_count = nx.number_of_nodes(G)
            edges_count = nx.number_of_edges(G)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            return  # Skip this dataset if there's an error

        # If the graph is too large, skip or sample
        if nodes_count > node_threshold:
            print(f"Graph is too large with {nodes_count} nodes")
            return

    report_file_features = 'reports/features_synthetic/' + graph_name + '.csv'
    writeToReport(report_file_features,
                  'degree, degree cent., max neighbor degree, min neighbor degree, avg neighbor degree, std neighbor degree, '
                  'eigenvector cent., closeness cent., harmonic cent., betweenness cent., '
                  'coloring largest first, coloring smallest last, coloring independent set, coloring random sequential,'
                  'coloring connected sequential dfs, coloring connected sequential bfs, edges within egonet,'
                  ' node clique number, number of cliques, clustering coef., square clustering coef., page rank, hubs value,'
                  ' triangles, core number, random ')

    start_total = datetime.datetime.now()
    metrics_count = 26
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    importance_array = np.zeros([metrics_count])
    iter_counter = 0


    walks = nx.generate_random_paths(G, iterations, walk_length)
    walks = list(walks)

    # Generate random subgraphs
    for walk in walks:
        try:
            subgraph_nodes = set()

            for n in walk:
                h = nx.ego_graph(G, n, radius=1)
                for node_h in h.nodes:
                    subgraph_nodes.add(node_h)

            subgraph_nodes = list(subgraph_nodes)
            # restrict to 50 nodes
            subgraph_nodes = subgraph_nodes[:50]
            H = G.subgraph(subgraph_nodes).copy()
            mapping = {old_label: new_label for new_label, old_label in enumerate(H.nodes())}
            H = nx.relabel_nodes(H, mapping)
            nodes_count_H = nx.number_of_nodes(H)
            edges_count_H = nx.number_of_edges(H)
            print('nodes count: ' + str(nodes_count_H))
            print('edges count: ' + str(edges_count_H))
            iter_counter += 1
        except ValueError as e:
            print(f"Warning: Skipping walk generation due to error: {e}")
            continue  # Skip this subgraph if there's an error in generating walks

        if nodes_count_H < 2:
            # skip if subgraph is too small
            continue

        start_time_features = datetime.datetime.now()

        start_time_degree_centrality = datetime.datetime.now()
        degree_centrality = nx.degree_centrality(H)
        end_time_degree_centrality = datetime.datetime.now()
        total_time_degree_centrality = (end_time_degree_centrality - start_time_degree_centrality)

        start_time_eigenvector_centrality = datetime.datetime.now()
        eigenvector_centrality = nx.eigenvector_centrality(H, max_iter=100, tol=1e-03)
        end_time_eigenvector_centrality = datetime.datetime.now()
        total_time_eigenvector_centrality = (end_time_eigenvector_centrality - start_time_eigenvector_centrality)

        start_time_closeness_centrality = datetime.datetime.now()
        closeness_centrality = nx.closeness_centrality(H)
        end_time_closeness_centrality = datetime.datetime.now()
        total_time_closeness_centrality = (end_time_closeness_centrality - start_time_closeness_centrality)

        start_time_harmonic_centrality = datetime.datetime.now()
        harmonic_centrality = nx.harmonic_centrality(H)
        end_time_harmonic_centrality = datetime.datetime.now()
        total_time_harmonic_centrality = (end_time_harmonic_centrality - start_time_harmonic_centrality)

        start_time_betweenness_centrality = datetime.datetime.now()
        betweenness_centrality = nx.betweenness_centrality(H)
        end_time_betweenness_centrality = datetime.datetime.now()
        total_time_betweenness_centrality = (end_time_betweenness_centrality - start_time_betweenness_centrality)

        start_time_coloring_lf = datetime.datetime.now()
        coloring_largest_first = nx.coloring.greedy_color(H, strategy='largest_first')
        end_time_coloring_lf = datetime.datetime.now()
        total_time_coloring_lf = (end_time_coloring_lf - start_time_coloring_lf)

        start_time_coloring_sl = datetime.datetime.now()
        try:
            coloring_smallest_last = nx.coloring.greedy_color(H, strategy='smallest_last')
        except KeyError as e:
            print(f"KeyError encountered while coloring node {e}. Skipping this coloring strategy.")
            continue
        end_time_coloring_sl = datetime.datetime.now()
        total_time_coloring_sl = (end_time_coloring_sl - start_time_coloring_sl)

        start_time_coloring_is = datetime.datetime.now()
        coloring_independent_set = nx.coloring.greedy_color(H, strategy='independent_set')
        end_time_coloring_is = datetime.datetime.now()
        total_time_coloring_is = (end_time_coloring_is - start_time_coloring_is)

        start_time_coloring_rs = datetime.datetime.now()
        coloring_random_sequential = nx.coloring.greedy_color(H, strategy='random_sequential')
        end_time_coloring_rs = datetime.datetime.now()
        total_time_coloring_rs = (end_time_coloring_rs - start_time_coloring_rs)

        start_time_coloring_dfs = datetime.datetime.now()
        coloring_connected_sequential_dfs = nx.coloring.greedy_color(H, strategy='connected_sequential_dfs')
        end_time_coloring_dfs = datetime.datetime.now()
        total_time_coloring_dfs = (end_time_coloring_dfs - start_time_coloring_dfs)

        start_time_coloring_bfs = datetime.datetime.now()
        coloring_connected_sequential_bfs = nx.coloring.greedy_color(H, strategy='connected_sequential_bfs')
        end_time_coloring_bfs = datetime.datetime.now()
        total_time_coloring_bfs = (end_time_coloring_bfs - start_time_coloring_bfs)

        start_time_node_clique_number = datetime.datetime.now()
        node_clique_number = nx.node_clique_number(H)
        end_time_node_clique_number = datetime.datetime.now()
        total_time_node_clique_number = (end_time_node_clique_number - start_time_node_clique_number)

        start_time_number_of_cliques = datetime.datetime.now()
        number_of_cliques = {n: sum(1 for c in nx.find_cliques(H) if n in c) for n in H}
        end_time_number_of_cliques = datetime.datetime.now()
        total_time_number_of_cliques = (end_time_number_of_cliques - start_time_number_of_cliques)

        start_time_clustering_coefficient = datetime.datetime.now()
        clustering_coefficient = nx.clustering(H)
        end_time_clustering_coefficient = datetime.datetime.now()
        total_time_clustering_coefficient = (end_time_clustering_coefficient - start_time_clustering_coefficient)

        start_time_square_clustering = datetime.datetime.now()
        square_clustering = nx.square_clustering(H)
        end_time_square_clustering = datetime.datetime.now()
        total_time_square_clustering = (end_time_square_clustering - start_time_square_clustering)

        start_time_average_neighbor_degree = datetime.datetime.now()
        average_neighbor_degree = nx.average_neighbor_degree(H)
        end_time_average_neighbor_degree = datetime.datetime.now()
        total_time_average_neighbor_degree = (end_time_average_neighbor_degree - start_time_average_neighbor_degree)

        start_time_hubs = datetime.datetime.now()
        hubs, authorities = nx.hits(H)
        end_time_hubs = datetime.datetime.now()
        total_time_hubs = (end_time_hubs - start_time_hubs)

        start_time_page_rank = datetime.datetime.now()
        page_rank = nx.pagerank(H)
        end_time_page_rank = datetime.datetime.now()
        total_time_page_rank = (end_time_page_rank - start_time_page_rank)

        start_time_core_number = datetime.datetime.now()
        core_number = nx.core_number(H)
        end_time_core_number = datetime.datetime.now()
        total_time_core_number = (end_time_core_number - start_time_core_number)

        end_time_features = datetime.datetime.now()
        total_time_features = (end_time_features - start_time_features)
        print('total time features: ', total_time_features)

        total_time_egonet = datetime.timedelta()
        total_time_triangles = datetime.timedelta()
        total_time_random = datetime.timedelta()


        comp_t_start = datetime.datetime.now()
        classes_count = np.random.randint(3, 5)
        comp_t_end = datetime.datetime.now() - comp_t_start

        start_time_lables = datetime.datetime.now()

        # Prepare the feature matrix
        X = np.zeros([nx.number_of_nodes(H), metrics_count])

        # Now fill in the feature matrix X for the random forest
        for i, v in enumerate(H):
            # Degree
            X[i][0] = H.degree(v)
            # Degree centrality
            X[i][1] = degree_centrality[i]
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
            X[i][4] = average_neighbor_degree[i]
            X[i][5] = std_neighbor_degree
            X[i][6] = eigenvector_centrality[i]
            X[i][7] = closeness_centrality[i]
            X[i][8] = harmonic_centrality[i]
            X[i][9] = betweenness_centrality[i]
            X[i][10] = coloring_largest_first[i]
            X[i][11] = coloring_smallest_last[i]
            X[i][12] = coloring_independent_set[i]
            X[i][13] = coloring_random_sequential[i]
            X[i][14] = coloring_connected_sequential_dfs[i]
            X[i][15] = coloring_connected_sequential_bfs[i]

            # Egonet edges
            start_time_egonet = datetime.datetime.now()
            egonet = nx.ego_graph(H, v, radius=1)
            edges_within_egonet = nx.number_of_edges(egonet)
            end_time_egonet = datetime.datetime.now()
            total_time_egonet += (end_time_egonet - start_time_egonet)

            X[i][16] = edges_within_egonet
            X[i][17] = node_clique_number[i]
            X[i][18] = number_of_cliques[i]
            X[i][19] = clustering_coefficient[i]
            X[i][20] = square_clustering[i]
            X[i][21] = page_rank[i]
            X[i][22] = hubs[i]

            start_time_triangles = datetime.datetime.now()
            X[i][23] = nx.triangles(H, v)
            end_time_triangles = datetime.datetime.now()
            total_time_triangles += (end_time_triangles - start_time_triangles)

            X[i][24] = core_number[i]

            # Random feature
            start_time_random = datetime.datetime.now()
            X[i][25] = np.random.normal(0, 1, 1)[0]
            end_time_random = datetime.datetime.now()
            total_time_random += (end_time_random - start_time_random)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        comp_t_start = datetime.datetime.now()
        K_feat = nx.adjacency_matrix(H).todense()

        # Then run SAF on K_feat
        W_init = np.random.rand(K_feat.shape[0], classes_count)

        W_final = self_assignment_flow(K_feat, W_init, max_iters=100, lr=0.005, s=0.5)#before 100 iterations
        # Convert assignment matrix to discrete labels
        # print("Final assignment matrix:", W_final)
        Y = np.argmax(W_final, axis=1)
        # Introduce a bit of noise in the labels
        Y_noisy = introduce_label_noise(Y, noise_fraction=0)
        print(f"Final labels for graph {graph_name}:", Y)
        end_time_lables = datetime.datetime.now()
        total_time_lables = (end_time_lables - start_time_lables)
        print('total time lables: ', total_time_lables)

        # Train a RandomForest on features X, with labels from self-assignment flow
        model = RandomForestClassifier()
        #model.fit(X, Y)
        model.fit(X, Y_noisy)
        arr = model.feature_importances_
        importance_array += arr
        comp_t_end = datetime.datetime.now() - comp_t_start + comp_t_end
        comp_t_total.append(comp_t_end)
        # Write features to file
        for x in X:
            writeToReport(report_file_features, list_to_str(x))
        writeToReport(report_file_features, '\n')

    importance_array_norm = importance_array / max(iter_counter, 1)
    arr_ordered = np.argsort(importance_array_norm)[::-1]
    print('Ranking: ' + str(arr_ordered))
    report_file = 'data/train_SAF/ranking_synthetic.csv'
    ranking_str = ''.join(str(m) + ',' for m in arr_ordered)
    writeToReport(report_file, graph_name + ',' + str(nodes_count) + ',' + str(edges_count) + ',' + ranking_str)

    report_file = 'data/train_SAF/importance_synthetic.csv'
    metrics_str = ''.join(str(np.round(m, 4)) + ',' for m in importance_array_norm)
    writeToReport(report_file, graph_name + ',' + str(nodes_count) + ',' + str(edges_count) + ',' + metrics_str)

    report_file = 'reports/computing_time.csv'
    writeToReport(report_file,
                  'graph_name , nodes_count , edges_count, degree, degree cent., max neighbor degree, min neighbor degree, avg neighbor degree, std neighbor degree, '
                  'eigenvector cent., closeness cent., harmonic cent., betweenness cent., '
                  'coloring largest first, coloring smallest last, coloring independent set, coloring random sequential,'
                  'coloring connected sequential dfs, coloring connected sequential bfs, edges within egonet,'
                  ' node clique number, number of cliques, clustering coef., square clustering coef., page rank, hubs value,'
                  ' triangles, core number, random')

    end = datetime.datetime.now()
    total_time = (end - start)

    script_name = os.path.basename(__file__)
    base_name = os.path.splitext(script_name)[0]

    report_file_time = f'time2/{base_name}.csv'

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
# include datasets to generate feature ranking
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
    f"{base_name},"
    f"{average_time_seconds:.4f},"
    f"{total_time_seconds:.4f}"  # total time for dataset in seconds
)
writeToReport(report_file_time, row_str)