import numpy as np
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
import ReadData
import os
import pickle
import torch_geometric
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
import util
from util import writeToReport
from torch_geometric.loader import DataLoader
from collections import OrderedDict
import datetime
from torch_geometric.datasets import (
    Planetoid, Amazon, Coauthor, Reddit, Reddit2, Yelp, AmazonProducts, Entities,
    CitationFull, NELL, Actor, GitHub, HeterophilousGraphDataset, Twitch, Airports
)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(26, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, 26)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        out1 = self.lin1(x)
        return out1


#######################################
# 1. Decide on which device to run:
#######################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


#######################################
# 2. Instantiate your model
#######################################
# Suppose you want hidden_channels=64, adjust if needed
model = GCN(hidden_channels=64)

# Load the saved model weights
# Make sure to map to the correct device:
model_name= 'model_SAF_adj_long_14_02'
model_path = f'model/{model_name}.pth'
model = torch.load(model_path, map_location=device)
model = model.to(device)  # Move model to GPU/CPU

model.eval()  # Set to evaluation mode


def predict(data):
    """
    data is a torch_geometric.data.Data object
    """
    with torch.no_grad():  # No gradient needed for inference
        # Move data to the same device as the model:
        data = data.to(device)
        out1 = model(data.x, data.edge_index, data.batch)
    return out1


# The rest of your code remains mostly the same,
# except that whenever you do inference (predict),
# ensure your data is on the correct device.

X_train = []
Y_train = []
X_test= []
Y_test = []
dataPath = 'data/train/'
dataset_length = 47
train_length = int(dataset_length * (1) / dataset_length)
X, Y = ReadData.readData('synthetic','test',dataPath)

print(X)
print(Y)

offset = 0
Y = Y[offset:offset+dataset_length]
X = X[offset:offset+dataset_length]
Y = np.reshape(Y,(len(Y),26,1))

batch = 1
y_all = []
y_all_test = []

for y in Y:
    tmp = []
    for f in y:
        tmp.append(f)
    y_all.append(tmp)


def add_attributes(dataset):
    data_list = []
    for i, data in enumerate(dataset):
        data.y = y_all[i]
        # create some node attributes:
        x_train = np.ones((data.num_nodes, 26), dtype=np.float32)
        x_train = torch.from_numpy(x_train)
        data.x = x_train
        data_list.append(data)
    return data_list


X_train = X[:train_length]
Y_train = Y[:train_length]
X_test = X[train_length:]
Y_test = Y[train_length:]


def load_graph(dataset_name, data_dir='./datasets', synthetic_dir='Synthetic/'):
    if 'Erdos' in dataset_name or 'Barabasi' in dataset_name:
        # Load synthetic graph
        print(f"Loading synthetic graph: {dataset_name}")
        try:
            G = torch_geometric.utils.from_networkx(
                pickle.load(open(f'{synthetic_dir}{dataset_name}', 'rb'))
            )
        except Exception as e:
            print(f"Failed to load synthetic graph {dataset_name}: {e}")
            return None
    else:
        # Load real-world dataset
        print(f"Loading real-world dataset: {dataset_name}")
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
            elif dataset_name == 'Actor':
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
                return None
            data = dataset[0]
            G = data  # torch_geometric.data.Data object
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            return None
    return G


dataset_list_test = []
graphs_list = []
for x in X:
    G = load_graph(x)
    if G is not None:
        graphs_list.append(G)

graphs_list = graphs_list[:dataset_length]
dataset_list = add_attributes(graphs_list)
dataset_list_test = dataset_list[train_length:]


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


report_file_features = 'Synthetic/' + 'test.csv'
writeToReport(report_file_features, 'Test Graph, Similarity, Ranking ')

k = 6
similarity = 0.0

test_loader = DataLoader(dataset_list_test, batch_size=1, shuffle=False)


def list_to_str(list_):
    return ",".join(str(l) for l in list_)


y_pred_list = []
y_pred_rank_list = []
y_test_list = []

time_now = datetime.datetime.now()
ranking_prediction_file = f'reports/rank_{model_name}'+'.csv'
ranking_test_file = 'reports/rank_test.csv'
importance_prediction_file = f'reports/importance{model_name}'+'.csv'
testing_time_file = 'reports/testing_time.csv'

writeToReport(ranking_prediction_file, 'graph, top 5 similarity, ranking')
writeToReport(ranking_test_file, 'graph, ranking')
writeToReport(importance_prediction_file, 'graph, importance')
writeToReport(testing_time_file, 'graph, time')


for i, data in enumerate(test_loader):
    test_time_start = datetime.datetime.now()

    #######################################
    # 3. Run predict on GPU/CPU
    #######################################
    output_list = predict(data).tolist()[0]

    test_time_end = datetime.datetime.now()
    test_time = (test_time_end - test_time_start)
    print(test_time)
    writeToReport(testing_time_file, X_test[i] + ',' + str(test_time))

    print('output list: ')
    print(output_list)

    y_pred = sorted(range(len(output_list)), key=lambda k: output_list[k], reverse=True)
    print('Y_pred: ' + str(output_list))
    print('Y_pred_rank: ' + str(y_pred))
    y_pred_list.append(output_list)
    y_pred_rank_list.append(y_pred)

    y_test = [y[0] for y in Y_test[i]]  # ground truth
    y_test_rank = sorted(range(len(output_list)), key=lambda k: y_test[k], reverse=True)
    print('Y_test ' + str(y_test))
    print('Y_test_rank ' + str(y_test_rank))
    y_test_list.append(y_test_rank)

    y_pred_top_k = y_pred[:k]
    y_test_top_k = y_test_rank[:k]
    jaccard_sim = jaccard(y_pred_top_k, y_test_top_k)
    print(jaccard_sim)
    similarity += jaccard_sim

    writeToReport(report_file_features, X_test[i] + ',' + str(jaccard_sim) + ',' + list_to_str(y_pred))
    writeToReport(ranking_prediction_file, X_test[i] + ',' + str(jaccard_sim) + ',' + list_to_str(y_pred))
    writeToReport(ranking_test_file, X_test[i] + ',' + list_to_str(y_test_rank))
    writeToReport(importance_prediction_file, X_test[i] + ',' + list_to_str(output_list))


print('avg sim: ')
print(similarity / len(Y_test))
