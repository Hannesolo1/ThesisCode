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
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import ReadData
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import random
import util
from util import writeToReport
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import export_text
from torch_geometric.datasets import Amazon, CitationFull, Entities, Twitch, Airports, Actor, GitHub
from torch_geometric.datasets import GNNBenchmarkDataset, AttributedGraphDataset, HeterophilousGraphDataset
import datetime
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, Coauthor
from collections import OrderedDict

# -----------------------
# Set device to CUDA if available
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # <-- Changed: set device to GPU if available
print("Using device:", device)  # <-- Changed: print which device is being used

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
        out = self.lin1(x)
        return out

def train():
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(device)  # <-- Changed: move the batch of data to GPU
        out1 = model(data.x, data.edge_index, data.batch)
        data.y = torch.reshape(data.y, (batch, 26))
        loss = criterion(out1, data.y)
        if (epoch % 50 == 0):
            print('Epoch: ' + str(epoch) + '   train loss: ' + str(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    correct = 0
    for i, data in enumerate(loader):
        data = data.to(device)  # <-- Changed: move the batch of data to GPU
        out1 = model(data.x, data.edge_index, data.batch)
        pred1 = out1.argmax(dim=1)
        data.y = data.y.transpose(0, 1)
        data.y = torch.reshape(data.y, (batch, 26))
        data.y = data.y.transpose(0, 1)
        correct += int((pred1 == data.y).sum())
    acc = int(correct) / len(loader.dataset)
    if (epoch == num_epochs - 1):
        test_results.append(pred1)
    return acc

X_train = []
Y_train = []
X_test = []
Y_test = []
X_val = []
Y_val = []

data_dir = "./datasets"
os.makedirs(data_dir, exist_ok=True)
dataPath = 'data/train_SAF_adj_long/'
dataset_length = 120
train_length = int(dataset_length * (10) / (12))
X, Y = ReadData.readData('synthetic', 'train', dataPath)
offset = 0
Y = Y[offset:offset + dataset_length]
X = X[offset:offset + dataset_length]
Y = np.reshape(Y, (len(Y), 26, 1))
print(X)
batch = 10
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
        x_train = np.ones((data.num_nodes, 26), dtype=np.float32)
        x_train = torch.from_numpy(x_train)
        data.x = x_train
        data_list.append(data)
    return data_list

y_all = np.array(y_all, dtype=np.float32)
y_all = torch.Tensor(y_all)
y_all = y_all.type(torch.FloatTensor)
dataset_list_train = []
dataset_list_test = []

graphs_list = [torch_geometric.utils.from_networkx(pickle.load(open('Synthetic/' + x, 'rb'))) for x in X]

X_train = X[:train_length]
Y_train = Y[:train_length]
X_test = X[train_length:]
Y_test = Y[train_length:]

graphs_list = graphs_list[:dataset_length]
dataset_list = add_attributes(graphs_list)
dataset_list_train = dataset_list[:train_length]
dataset_list_test = dataset_list[train_length:]
train_dataset = dataset_list_train[:train_length]
test_dataset = dataset_list_test[:(dataset_length - train_length)]

train_loader = DataLoader(dataset_list_train, batch_size=batch, shuffle=False)
test_loader = DataLoader(dataset_list_test, batch_size=batch, shuffle=False)

for i in train_loader:
    print(i)

dataSet = 'pattern'

model = GCN(64).to(device)  # <-- Changed: move the model to GPU
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss().to(device)  # <-- Changed: move the loss function to GPU
test_results = []
num_epochs = 20000  # normally 20000

for epoch in range(1, num_epochs):
    train()
test_acc = test(test_loader)

def predict(data):
    model_path = 'model/model_SAF_adj_long_14_02.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)
    model.eval()
    data = data.to(device)  # <-- Changed: move the data to GPU before running prediction
    out1 = model(data.x, data.edge_index, data.batch)
    return out1

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

report_file_features = 'Synthetic/' + 'test.csv'
report_file_time = 'Synthetic/' + 'time.txt'
writeToReport(report_file_features, 'Test Graph, Similaritiy, Ranking ')
k = 6
similarity = 0.0
test_loader = DataLoader(dataset_list_test, batch_size=1, shuffle=False)

def list_to_str(list):
    str_list = ''
    for l in list:
        str_list += str(l) + ','
    return str_list

y_pred_list = []
y_pred_rank_list = []
y_test_list = []

time = datetime.datetime.now()
ranking_prediction_file = 'reports/rank_pred_' + str(time) + '.csv'
ranking_test_file = 'reports/rank_test_' + str(time) + '.csv'
importance_prediction_file = 'reports/importance_pred_' + str(time) + '.csv'

writeToReport(ranking_prediction_file, 'graph, top 5 similarity, ranking')
writeToReport(ranking_test_file, 'graph, ranking')
writeToReport(importance_prediction_file, 'graph, importance')

for i, data in enumerate(test_loader):
    test_time_start = datetime.datetime.now()
    data = data.to(device)  # <-- Changed: move the data to GPU within the test loop
    output_list = predict(data).tolist()[0]
    test_time_end = datetime.datetime.now()
    test_time = (test_time_end - test_time_start)

    print(test_time)
    print('output list: ')
    print(output_list)

    y_pred = sorted(range(len(output_list)), key=lambda k: output_list[k], reverse=True)
    print('Y_pred: ' + str(output_list))
    print('Y_pred_rank: ' + str(y_pred))
    y_pred_list.append(output_list)
    y_pred_rank_list.append(y_pred)

    y_test = [y[0] for y in Y_test[i]]
    y_test_rank = sorted(range(len(output_list)), key=lambda k: y_test[k], reverse=True)
    print('Y_test' + str(y_test))
    print('Y_test_rank' + str(y_test_rank))
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
    writeToReport(report_file_time, str(test_time))

print('avg sim: ')
print(similarity / len(Y_test))
