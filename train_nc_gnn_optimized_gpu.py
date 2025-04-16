import os
import numpy as np
import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    Planetoid, Amazon, Coauthor, Reddit, Yelp, AmazonProducts, Entities,
    CitationFull, NELL, Actor, GitHub, HeterophilousGraphDataset, Airports, AttributedGraphDataset
)
from collections import Counter
import random
import datetime
import pickle
import csv

# -----------------------
# 1) Set random seeds for reproducibility
# -----------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------------------
# 2) Choose device (CPU or GPU)
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # <-- Changed: set device to GPU if available
print("Using device:", device)  # <-- Changed: print which device is being used

# Define constants
data_dir = "./datasets"
os.makedirs(data_dir, exist_ok=True)
hidden_layers = 64
optimal_k = 6  # You can adjust this as needed

# Read feature rankings from 'rank_predicted.csv' and store in a dictionary
rankings_dict = {}
with open('reports/rank_model_SAF_adj_long_14_02.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        if row:
            graph_name = row[0]
            # Assuming feature rankings start from column index 2
            ranking = [int(col) for col in row[2:] if col != '']
            rankings_dict[graph_name] = ranking

graph_name_list = list(rankings_dict.keys())

def file_exists_case_insensitive(directory, filename):
    filename_lower = filename.lower()
    for f in os.listdir(directory):
        if f.lower() == filename_lower:
            return os.path.join(directory, f)
    return "None"

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(input_features, hidden_layers)
        self.conv2 = GCNConv(hidden_layers, hidden_layers)
        self.conv3 = GCNConv(hidden_layers, num_classes)

    def forward(self, data, X):
        edge_index = data.edge_index
        x = self.conv1(X, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        output = self.conv3(x, edge_index)
        return output

# Training function
def train_node_classifier(model, graph, X, optimizer, criterion, n_epochs=3000, verbose=False):
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph, X)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            acc = eval_node_classifier(model, graph, X, graph.val_mask)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {acc:.4f}')
    return model

# Evaluation function
def eval_node_classifier(model, graph, X, mask):
    model.eval()
    with torch.no_grad():
        out = model(graph, X)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == graph.y[mask]).sum()
        acc = correct.item() / mask.sum().item()
    return acc

for graph_name in graph_name_list:
    filename = file_exists_case_insensitive('data/synthetic/', graph_name + '.csv')
    if filename == "None":
        print(f"Graph {graph_name} features not calculated. Skipping...")
        continue
    if graph_name not in os.listdir('data/synthetic/') and not os.path.exists(filename):
        print(f"Graph {graph_name} features not calculated. Skipping...")
        continue
    print(f'\nProcessing graph: {graph_name}')
    ranking = rankings_dict[graph_name]

    # Initialize lists to store accuracies
    acc_metrics_list = []
    acc_random_list = []
    acc_vanilla_list = []
    acc_all_features_list = []

    for iteration in range(1, 4):
        print(f'\nIteration {iteration} for graph: {graph_name}')
        metrics_count = 26  # Adjust based on your dataset
        random_features_indices = random.sample(range(metrics_count), optimal_k)

        # Load dataset based on graph name
        if (graph_name == 'Cora'):
            classes_count = 7
            dataset = Planetoid(root=data_dir, name=graph_name)
        elif (graph_name == 'CiteSeer'):
            classes_count = 6
            dataset = Planetoid(root=data_dir, name=graph_name)
        elif (graph_name == 'PubMed'):
            classes_count = 3
            dataset = Planetoid(root=data_dir, name=graph_name)
        elif (graph_name == 'Photo'):
            classes_count = 8
            dataset = Amazon(root=data_dir, name=graph_name)
        elif (graph_name == 'Computers'):
            classes_count = 10
            dataset = Amazon(root=data_dir, name=graph_name)
        elif (graph_name == 'CS'):
            classes_count = 15
            dataset = Coauthor(root=data_dir, name=graph_name)
        elif (graph_name == 'Physics'):
            classes_count = 5
            dataset = Coauthor(root=data_dir, name=graph_name)
        elif (graph_name == 'Cora_ML'):
            classes_count = 7
            dataset = CitationFull(root=data_dir, name=graph_name)
        elif (graph_name == 'DBLP'):
            classes_count = 4
            dataset = CitationFull(root=data_dir, name=graph_name)
        elif (graph_name == 'NELL'):
            dataset = NELL(root=data_dir)
            classes_count = dataset.num_classes
        elif (graph_name == 'Reddit'):
            dataset = Reddit(root=data_dir)
            classes_count = dataset.num_classes
        elif (graph_name == 'Yelp'):
            dataset = Yelp(root=data_dir)
            classes_count = dataset.num_classes
        elif (graph_name == 'AmazonProducts'):
            dataset = AmazonProducts(root=data_dir)
            classes_count = dataset.num_classes
        elif (graph_name == 'AIFB'):
            dataset = Entities(root=data_dir, name=graph_name)
            classes_count = dataset.num_classes
        elif (graph_name == 'MUTAG'):
            dataset = Entities(root=data_dir, name=graph_name)
            classes_count = dataset.num_classes
        elif (graph_name == 'Actor'):
            dataset = Actor(root=data_dir)
            classes_count = dataset.num_classes
        elif (graph_name == 'GitHub'):
            dataset = GitHub(root=data_dir)
            classes_count = dataset.num_classes
        elif (graph_name == 'Roman-empire'):
            classes_count = 18
            dataset = HeterophilousGraphDataset(root=data_dir, name=graph_name)
        elif (graph_name == 'Amazon-ratings'):
            classes_count = 5
            dataset = HeterophilousGraphDataset(root=data_dir, name=graph_name)
        elif (graph_name == 'Minesweeper'):
            dataset = HeterophilousGraphDataset(root=data_dir, name=graph_name)
            classes_count = dataset.num_classes
        elif (graph_name == 'Tolokers'):
            classes_count = 2
            dataset = HeterophilousGraphDataset(root=data_dir, name=graph_name)
        elif (graph_name == 'Questions'):
            classes_count = 2
            dataset = HeterophilousGraphDataset(root=data_dir, name=graph_name)
        elif (graph_name == 'USA' or graph_name == 'Brazil' or graph_name == 'Europe'):
            classes_count = 4
            dataset = Airports(root=data_dir, name=graph_name)
        elif (graph_name == 'Wiki'):
            classes_count = 17
            dataset = AttributedGraphDataset(root=data_dir, name=graph_name)
        elif ('house' in graph_name):
            classes_count = 4
        elif ('star' in graph_name):
            classes_count = 3
        elif ('grid' in graph_name):
            classes_count = 4
        elif ('path' in graph_name):
            classes_count = 3
        elif ('cycle' in graph_name):
            classes_count = 2
        else:
            print(f"Graph {graph_name} not found. Skipping...")
            continue

        # Load the data
        if 'house' in graph_name or 'star' in graph_name or 'grid' in graph_name or 'path' in graph_name or 'cycle' in graph_name:
            print(f'Loading synthetic graph: {graph_name}')
            data = torch_geometric.utils.from_networkx(pickle.load(open('Synthetic/' + graph_name, 'rb')))
        else:
            data = dataset[0]

        # -----------------------
        # 3) Move your data object to the GPU
        # -----------------------
        data = data.to(device)  # <-- Changed: move PyG data to GPU

        # Load features and labels
        filename = file_exists_case_insensitive('data/synthetic/', graph_name + '.csv')
        X_data = np.loadtxt(filename, delimiter=',', dtype=str, skiprows=1)

        # Check if the last column header contains 'Target' or similar
        if 'Target' in X_data[0][-1]:
            Y = X_data[1:, -1].astype(np.float64).astype(np.int64)
            X = X_data[1:, :-1].astype(np.float32)
        else:
            Y = X_data[:, -1].astype(np.float64).astype(np.int64)
            X = X_data[:, :-1].astype(np.float32)

        metrics_count = X.shape[1]

        # Prepare features
        k = optimal_k
        k_ranking = ranking[:k]
        X_metrics = X[:, k_ranking]
        X_random = X[:, random_features_indices]
        X_vanilla = np.ones((len(X), metrics_count), dtype=np.float32)
        X_all = X  # Use all available features

        # Convert to torch tensors
        X_metrics_torch = torch.from_numpy(X_metrics).float()
        X_random_torch = torch.from_numpy(X_random).float()
        X_vanilla_torch = torch.from_numpy(X_vanilla).float()
        X_all_torch = torch.from_numpy(X_all).float()
        data.y = torch.from_numpy(Y).long().to(device)  # <-- Changed: move `y` labels to GPU

        # -----------------------
        # 4) Also move feature tensors to GPU
        # -----------------------
        X_metrics_torch = X_metrics_torch.to(device)  # <-- Changed
        X_random_torch = X_random_torch.to(device)    # <-- Changed
        X_vanilla_torch = X_vanilla_torch.to(device)  # <-- Changed
        X_all_torch = X_all_torch.to(device)          # <-- Changed

        # Split data
        split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
        data = split(data)

        label_counts = Counter(data.y[data.train_mask].cpu().numpy())
        total_count = sum(label_counts.values())
        class_weights = torch.tensor([total_count / label_counts[i] for i in range(len(label_counts))],
                                     dtype=torch.float).to(device)

        # Define criterion
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)  # <-- Changed: move criterion to GPU

        # Train GCN on top-k features
        gcn_metrics = GCN(input_features=X_metrics_torch.shape[1], num_classes=classes_count)
        gcn_metrics = gcn_metrics.to(device)  # <-- Changed: move model to GPU
        optimizer_metrics = torch.optim.Adam(gcn_metrics.parameters(), lr=0.001, weight_decay=5e-4)
        gcn_metrics = train_node_classifier(gcn_metrics, data, X_metrics_torch, optimizer_metrics, criterion,
                                            verbose=False)
        test_acc_metrics = eval_node_classifier(gcn_metrics, data, X_metrics_torch, data.test_mask)
        acc_metrics_list.append(test_acc_metrics)

        # Train GCN on random features
        gcn_random = GCN(input_features=X_random_torch.shape[1], num_classes=classes_count)
        gcn_random = gcn_random.to(device)  # <-- Changed
        optimizer_random = torch.optim.Adam(gcn_random.parameters(), lr=0.001, weight_decay=5e-4)
        gcn_random = train_node_classifier(gcn_random, data, X_random_torch, optimizer_random, criterion, verbose=False)
        test_acc_random = eval_node_classifier(gcn_random, data, X_random_torch, data.test_mask)
        acc_random_list.append(test_acc_random)

        # Train GCN on vanilla features
        gcn_vanilla = GCN(input_features=X_vanilla_torch.shape[1], num_classes=classes_count)
        gcn_vanilla = gcn_vanilla.to(device)  # <-- Changed
        optimizer_vanilla = torch.optim.Adam(gcn_vanilla.parameters(), lr=0.001, weight_decay=5e-4)
        gcn_vanilla = train_node_classifier(gcn_vanilla, data, X_vanilla_torch, optimizer_vanilla, criterion,
                                            verbose=False)
        test_acc_vanilla = eval_node_classifier(gcn_vanilla, data, X_vanilla_torch, data.test_mask)
        acc_vanilla_list.append(test_acc_vanilla)

        # Train GCN on all features
        gcn_all = GCN(input_features=X_all_torch.shape[1], num_classes=classes_count)
        gcn_all = gcn_all.to(device)  # <-- Changed
        optimizer_all = torch.optim.Adam(gcn_all.parameters(), lr=0.001, weight_decay=5e-4)
        gcn_all = train_node_classifier(gcn_all, data, X_all_torch, optimizer_all, criterion, verbose=False)
        test_acc_all = eval_node_classifier(gcn_all, data, X_all_torch, data.test_mask)
        acc_all_features_list.append(test_acc_all)

        # Print iteration results
        print(f'\nIteration {iteration} Results:')
        print(f'Test Accuracy with top-{k} features: {test_acc_metrics:.4f}')
        print(f'Test Accuracy with random features: {test_acc_random:.4f}')
        print(f'Test Accuracy with vanilla features: {test_acc_vanilla:.4f}')
        print(f'Test Accuracy with all features: {test_acc_all:.4f}')

    # Compute average accuracies and differences
    avg_acc_metrics = np.mean(acc_metrics_list)
    avg_acc_random = np.mean(acc_random_list)
    avg_acc_vanilla = np.mean(acc_vanilla_list)
    avg_acc_all = np.mean(acc_all_features_list)

    # Compute standard deviations
    std_acc_metrics = np.std(acc_metrics_list)
    std_acc_random = np.std(acc_random_list)
    std_acc_vanilla = np.std(acc_vanilla_list)
    std_acc_all = np.std(acc_all_features_list)

    diff_metrics_random = avg_acc_metrics - avg_acc_random
    diff_metrics_vanilla = avg_acc_metrics - avg_acc_vanilla
    diff_metrics_all = avg_acc_metrics - avg_acc_all
    diff_all_random = avg_acc_all - avg_acc_random
    diff_all_vanilla = avg_acc_all - avg_acc_vanilla

    # Print summary
    print(f'\nSummary for graph: {graph_name}')
    print(f'Average Test Accuracies over {iteration} iterations:')
    print(f' - Top-{k} features: {avg_acc_metrics:.4f}')
    print(f' - Random features: {avg_acc_random:.4f}')
    print(f' - Vanilla features: {avg_acc_vanilla:.4f}')
    print(f' - All features: {avg_acc_all:.4f}')
    print('\nAccuracy Differences:')
    print(f' - Top-{k} - Random: {diff_metrics_random:.4f}')
    print(f' - Top-{k} - Vanilla: {diff_metrics_vanilla:.4f}')
    print(f' - Top-{k} - All Features: {diff_metrics_all:.4f}')
    print(f' - All Features - Random: {diff_all_random:.4f}')
    print(f' - All Features - Vanilla: {diff_all_vanilla:.4f}')

    report_file = 'accuracy_reports/accuracy_SAF_adj_long_14_02.csv'
    top_k_features_str = ';'.join(str(f) for f in ranking[:k])

    if not os.path.exists(report_file):
        with open(report_file, 'w') as f:
            f.write(
                'Graph,Top-k Acc,Random Acc,Vanilla Acc,All Features Acc,Top-k - Random,Top-k - Vanilla,Top-k - All,All - Random,All - Vanilla,Top-k Features\n')

    with open(report_file, 'a') as f:
        f.write(f'{graph_name},{avg_acc_metrics:.4f},{avg_acc_random:.4f},{avg_acc_vanilla:.4f},{avg_acc_all:.4f},'
                f'{diff_metrics_random:.4f},{diff_metrics_vanilla:.4f},{diff_metrics_all:.4f},'
                f'{diff_all_random:.4f},{diff_all_vanilla:.4f},{top_k_features_str}\n')

    # Create the standard deviation report file if it doesn't exist
    std_report_file = 'accuracy_reports_std/accuracy_std_SAF_adj_long_14_02.csv'

    if not os.path.exists(std_report_file):
        with open(std_report_file, 'w') as f:
            f.write(
                'Graph,Top-k STD,Random STD,Vanilla STD,All Features STD\n'
            )

    with open(std_report_file, 'a') as f:
        f.write(f'{graph_name},{std_acc_metrics:.4f},{std_acc_random:.4f},'
                f'{std_acc_vanilla:.4f},{std_acc_all:.4f}\n')
