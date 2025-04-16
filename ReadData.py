import numpy as np
import csv
import operator
import sys
import os
from sklearn.datasets import load_svmlight_file

def readData(dataset, type, path):
	if dataset == 'Cora':
		return readDataCora(type, path)
	if dataset == 'cora_all':
		return readDataCoraAll(type, path)
	if dataset == 'CiteSeer':
		return readDataCiteSeer(type, path)
	if dataset == 'PubMed':
		return readDataPubmed(type, path)
	if dataset == 'Photo':
		return readDataPhoto(type, path)
	if dataset == 'amazon_photo_all':
		return readDataAmazonPhotoAll(type, path)
	if dataset == 'amazon_computers_all':
		return readDataAmazonComputersAll(type, path)
	if dataset == 'predictions_all':
		return readDataPredictionsAll(type, path)
	if dataset == 'predictions':
		return readDataPredictions(type, path)
	if dataset == 'pattern':
		return readDataPattern(type, path)
	if dataset == 'pattern_all':
		return readDataPatternAll(type, path)
	if dataset == 'unsupervised':
		return readDataUnsupervised(type, path)
	if dataset == 'synthetic':
			return readDataSynthetic(type, path)
	if dataset == 'graph_classification':
		return readDataGraphClassification(type, path)
	if dataset == 'USA':
		return readDataUSA(type, path)
	if dataset == 'house_600_12_0.012_4' or dataset == 'house_20_18_1_2' or dataset == 'house_60_5_1_2':
		return readDataSynthetic(dataset,type,path)

	raise Exception('Name of dataset unknown: ' + dataset)


def readDataUnsupervised(type, path="data/unsupervised/"):
	if (type == 'train'):
		filename = os.path.join(path, "unsupervised", "unsupervised.train")
	#if (type == 'test'):
	#	filename = os.path.join(path, "pattern", "pattern.test")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, 2:]
	#X = X[:, 0:3]

	return Y

def readDataSyntheticAll(dataset, type, path="data/train/"):
	if (type == 'train'):
		filename = os.path.join(path, dataset + ".train")
	if (type == 'test'):
		filename = os.path.join(path, dataset, ".test")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, -1]
	X = X[:, :-1]
	#Y = np.array(Y)
	#print(Y)
	#Y = Y.astype(np.float32)

	return X,Y

def readDataSynthetic(type, path="data/train/"):
	if (type == 'train'):
		filename = os.path.join(path, 'synthetic' + ".train")
	if (type == 'test'):
		filename = os.path.join(path, 'synthetic'+ ".test")

	X = np.loadtxt(filename, delimiter=',', dtype=str)
	Y = X[:, 3:]
	X = X[:, 0]

	if Y.shape[1] > 0 and np.all(Y[:, -1] == ''):
		Y = Y[:, :-1]  # Exclude the last column

	Y = np.array(Y)
	#print("Y before conversion:", Y)
	Y = Y.astype(np.float32)

	return X,Y


def readDataGraphClassification(type, path="data/synthetic/"):
	if (type == 'train'):
		filename = os.path.join(path, 'synthetic', 'training_graphs' + ".train")
	#if (type == 'test'):
	#	filename = os.path.join(path, "pattern", "pattern.test")

	X = np.loadtxt(filename, delimiter=',', dtype=str)
	Y = X[:, 3:]
	X = X[:, 0]
	Y = np.array(Y)
	#print(Y)
	Y = Y.astype(np.float32)

	return X,Y

def readDataPattern(type, path="data/pattern/"):
	if (type == 'train'):
		filename = os.path.join(path, "pattern", "pattern.train")
	if (type == 'test'):
		filename = os.path.join(path, "pattern", "pattern.test")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, 4:]
	X = X[:, 0:3]

	return X, Y

def readDataPatternAll(type, path="data/pattern/"):
	if (type == 'train'):
		filename = os.path.join(path, "pattern", "pattern_all.train")
	#if (type == 'test'):
	#	filename = os.path.join(path, "pattern", "pattern.test")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, 4:]
	X = X[:, 0:3]

	return Y

def readDataCora(type, path="data/Cora/"):
	if (type == 'train'):
		filename = os.path.join(path, "Cora", "Cora.train")
	#if (type == 'test'):
	#		filename = os.path.join(path, "cora", "cora.test")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, -1]
	X = X[:, :-1]

	return X, Y

def readDataCoraAll(type, path="data/Cora/"):
	if (type == 'train'):
		filename = os.path.join(path, "cora", "cora_all.train")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, -1]
	X = X[:, 1:-1]

	return X, Y

def readDataCiteSeer(type, path="data/CiteSeer/"):
	if (type == 'train'):
		filename = os.path.join(path, "CiteSeer", "CiteSeer.train")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, -1]
	X = X[:, :-1]

	return X, Y

def readDataUSA(type, path="data/USA/"):
	if (type == 'train'):
		filename = os.path.join(path, "USA", "USA.train")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, -1]
	X = X[:, :-1]

	return X, Y



def readDataPubmed(type, path="data/PubMed/"):
	if (type == 'train'):
		filename = os.path.join(path, "PubMed", "PubMed.train")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, -1]
	X = X[:, :-1]

	return X, Y

def readDataPhoto(type, path="data/Photo/"):
	if (type == 'train'):
		filename = os.path.join(path, "Photo", "Photo.train")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, -1]
	X = X[:, :-1]

	return X, Y

def readDataAmazonComputersAll(type, path="data/amazon_computers/"):
	if (type == 'train'):
		filename = os.path.join(path, "amazon_computers", "amazon_computers_all.train")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, -1]
	X = X[:, 1:-1]

	return X, Y


def readDataPredictionsAll(type, path="data/"):
	if (type == 'train'):
		filename = os.path.join(path, "predictions_all", "predictions_all.train")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	Y = X[:, -1]
	X = X[:, :-1]

	return X, Y


def readDataPredictions(type, path="data/"):
	if (type == 'train'):
		filename = os.path.join(path, "predictions", "predictions.train")
	if (type == 'test'):
		filename = os.path.join(path, "predictions", "predictions.test")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float64)
	Y = X[:, -1]
	X = X[:, 0:-1]

	return X, Y
