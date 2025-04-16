# Graph Feature Importance Learning Framework

This repository contains code and data accompanying the bachelor thesis:

**"Unsupervised Feature Importance Ranking in Graph Structured Data"**

The project implements various methods to compute graph features from both synthetic and real-world datasets, learn feature importance with graph neural networks (GNNs), and rank these features using multiple strategies. Methods include random subgraph sampling, self-assignment flow (SAF), community detection (using the Leiden algorithm), clustering (K-Means), and Layer-wise Relevance Propagation (LRP).

## Repository Overview

The repository is organized into several modules that cover:
- **Graph Feature Computation:** Calculating node-level and graph-level features.
- **Feature Ranking:** Ranking features based on their importance using different strategies.
- **Graph Neural Networks:** Both for feature learning and node classification.
- **Synthetic Graph Generation:** Creating synthetic graphs from base models and motifs.
- **Data Management:** Loading data, merging computed values into training files, and auxiliary utilities.
- **Explanation Methods:** Using LRP to explain model predictions.

Each module reads data from and writes outputs to specific directories. For example, synthetic graphs are read from and saved to the `Synthetic/` folder, while computed feature CSV files are written in `data/synthetic/` or `reports/features_synthetic/`.

## Detailed File Descriptions

### Feature Computation
- **compute_features_multithreaded_SyntheticGraphs.py**  
  *Purpose:* Calculates features for synthetic graphs in parallel.  
  *Inputs:* Synthetic graphs (pickle files) from the `Synthetic/` folder.  
  *Outputs:* CSV reports with computed features (e.g., degree, centrality measures, clustering coefficients, coloring strategies) are saved under `data/synthetic/`.  

- **compute_features_RealWorldGraphs.py**  
  *Purpose:* Computes similar features for real-world graph datasets.  
  *Inputs:* Real-world graphs (e.g., Cora, PubMed) obtained via PyTorch Geometric from their respective online repositories/data directories.  
  *Outputs:* Feature CSV files for each processed graph are saved under `data/synthetic/` (or another designated directory).

### Feature Learning & Ranking
- **feature_learning_gnn_gpu.py**  
  *Purpose:* Implements a two-layer GCN that learns to associate graph structure with feature rankings by training on synthetic graphs with corresponding feature rankings. Runs on GPU.  
  *Inputs:* Feature rankings (e.g. from `data/train/synthetic.train`) and synthetic graph structures from the `Synthetic/` folder.  
  *Outputs:* A trained model is saved to disk under the `model/` folder; training logs and report files are written to `reports/`.

- **generate_features_ranking_GNN_GSM.py**  
  *Purpose:* Ranks features using a Gradient Saliency Map (GSM) GCN-based approach combined with random labeling.  
  *Inputs:* Graphs read from the `Synthetic/` folder and features from CSV files in `data/synthetic/`.  
  *Outputs:* Ranking and importance scores for synthetic training graphs are saved in directories such as `data/train_GNN_GSM/ranking_synthetic.csv` and `data/train_GNN_GSM/importance_synthetic.csv`. These are then used by the feature learning GNN.

- **generate_features_ranking_GNN_GSM_Leiden.py**  
  *Purpose:* GSM feature ranking that incorporates community detection using the Leiden algorithm for labeling the synthetic graphs.  
  *Inputs & Outputs:* Same as above, with output directory paths such as `data/train_GSM_Leiden/ranking_synthetic.csv` and `data/train_GSM_Leiden/importance_synthetic.csv`.

- **generate_features_ranking_GNN_LRP.py**  
  *Purpose:* Utilizes Layer-wise Relevance Propagation (LRP) for feature ranking. Synthetic graphs are labeled with random labels. The LRP method is then applied to compute feature importances.  
  *Key Class:* *LRPExplainer* (see below).  
  *Inputs & Outputs:* As above, with outputs saved in directories like `data/train_LRP/ranking_synthetic.csv` and `data/train_LRP/importance_synthetic.csv`.

- **generate_features_ranking_K_Means.py**  
  *Purpose:* K-Means clustering is applied to label the synthetic graphs, and then the Random Forest feature ranking method is used to compute feature importances.  
  *Inputs & Outputs:* Similar to the other ranking modules, with output directory paths such as `data/train_K_Means/ranking_synthetic.csv` and `data/train_K_Means/importance_synthetic.csv`.

- **generate_features_ranking_Leiden.py** and **generate_features_ranking_Leiden_Modified.py**  
  *Purpose:* Use the Leiden community detection algorithm to label synthetic graphs. They then use the Random Forest feature ranking method to compute feature importances.  
  *Difference:* The modified version adjusts for small communities by merging them according to edge connectivity.  
  *Inputs & Outputs:* Outputs are saved to directories like `data/train_Leiden/ranking_synthetic.csv` or `data/train_Leiden_Mod/ranking_synthetic.csv` along with their respective importance files.

- **generate_features_ranking_Random.py**  
  *Purpose:* Uses random labeling for synthetic graphs and Random Forest for feature ranking.  
  *Inputs & Outputs:* Output directory paths such as `data/train_Random/ranking_synthetic.csv` and `data/train_Random/importance_synthetic.csv`.

- **generate_features_ranking_SAF.py**  
  *Purpose:* Implements the Self-Assignment Flow (SAF) algorithm for synthetic graph labeling and uses Random Forest for feature ranking.  
  *Inputs & Outputs:* Output directory paths such as `data/train_SAF/ranking_synthetic.csv` and `data/train_SAF/importance_synthetic.csv`.

### Synthetic Graph Generation
- **generate_synthetic_graphs_testing.py** and **generate_synthetic_graphs_training.py**  
  *Purpose:* Generate synthetic graphs based on different base models (Barabasi-Albert or Erdos-Renyi) and motifs (path, house, grid, star, cycle, with optional noisy edges).  
  *Inputs:* Parameters defined within the scripts (e.g., node count, motif count, motif shape, noisy edges).  
  *Outputs:* Synthetic graph pickle files are saved in the `Synthetic/` folder. They have a title structure: `Basegraph_NodeCount_EdgeCount_MotifShape_MotifCount_NoisyEdgesCount_Index.pickle` (e.g., `Barabasi_100_200_path_5_0_0.pickle`).  
  *Additional Details:* The training script creates the graphs needed for training from the file `data/train/training_graphs.csv`, while the testing script uses the provided parameters.

### Explanation and Data Management
- **LRPExplainer.py**  
  *Purpose:* Provides a minimal implementation of LRP for a 2-layer GCN.  
  *Key Methods:*  
    - `forward`: Performs a forward pass while storing intermediate activations (input to pre-softmax logits).  
    - `attribute`: Distributes relevance scores backward to compute feature importance.  
  *Inputs:* Node feature matrices, graph edge index, and true labels (from computed data).  
  *Outputs:* Relevance scores that inform ranking, used by `generate_features_ranking_GNN_LRP.py`.

- **moveDataToTrain.py**  
  *Purpose:* Merges feature rankings from the various generate_features_ranking_... modules into the training file.  
  *Inputs:* A training file (e.g., from `data/train_SAF/ranking_synthetic.csv`).  
  *Outputs:* An updated training file is saved to `data/train_SAF/synthetic.train` (or a similar directory based on the experiment).

- **ReadData.py**  
  *Purpose:* Contains functions for reading graph datasets from CSV files or other formats.  
  *Inputs:* Dataset names and file paths, e.g., synthetic graphs from `data/train/` or real-world datasets from their standard directories.  
  *Outputs:* Returns numpy arrays for feature matrices and corresponding labels.

### Model Testing and Training
- **test_model.py**  
  *Purpose:* Loads a pre-trained GCN model (from `feature_learning_gnn_gpu.py`) and applies it to test graphs to predict feature rankings.  
  *Inputs:* A PyTorch Geometric data object representing the graph and a model file from the `model/` folder.  
  *Outputs:* Predicted rankings and importance scores are written to `reports/rank_predicted/` and `reports/importance_predicted/`.

- **train_nc_gnn_optimized_gpu.py**  
  *Purpose:* An optimized training script for node classification using a GCN on GPU. It uses feature rankings (e.g., from `reports/rank_predicted/`) for training.  
  *Inputs:* Graph datasets and feature rankings from directories like `data/synthetic/` or from PyTorch Geometric.  
  *Outputs:* Training and evaluation results (accuracy, loss metrics) are displayed in the console and saved into result files under `accuracy_reports/` or `reports/`.

## Summary

This framework allows you to generate synthetic graphs, compute their features, rank these features via multiple methods (GNN-GSM, LRP, K-Means, Leiden-based, Random, or SAF), and then use the ranked features to train graph neural networks for feature learning and node classification tasks. Data flows from synthetic graph generation to feature computation, then to feature ranking, and finally to model training and testing. The repository is designed to support unsupervised feature importance ranking in graph structured data with clear directory structures for inputs and outputs.

## Execution Order

To effectively use the framework, follow this recommended sequence:

1. **Generate Synthetic Graphs:**
   - For training graphs, run:
     ```bash
     python generate_synthetic_graphs_training.py
     ```
   - For testing graphs, run:
     ```bash
     python generate_synthetic_graphs_testing.py
     ```
   *Output:* Synthetic graph pickle files are created and saved in the `Synthetic/` folder.

2. **Compute Graph Features:**
   - For synthetic graphs, execute:
     ```bash
     python compute_features_multithreaded_SyntheticGraphs.py
     ```
   - For real-world datasets, run:
     ```bash
     python compute_features_RealWorldGraphs.py
     ```
   *Output:* CSV reports with computed features are saved under `data/synthetic/` and/or `reports/features_synthetic/`.

3. **Generate Feature Rankings:**
   - Choose one or more ranking methods depending on your experimental design:
     ```bash
     python generate_features_ranking_GNN_GSM.py
     python generate_features_ranking_GNN_GSM_Leiden.py
     python generate_features_ranking_GNN_LRP.py
     python generate_features_ranking_K_Means.py
     python generate_features_ranking_Leiden.py
     python generate_features_ranking_Random.py
     python generate_features_ranking_SAF.py
     ```
   *Output:* Ranking and importance scores are saved to their respective directories (e.g., `data/train_GNN_GSM/`, `data/train_LRP/`, etc.).

4. **Merge Rankings into Training Files:**
   - Run:
     ```bash
     python moveDataToTrain.py
     ```
   *Output:* An updated training file (e.g., `data/train_SAF/synthetic.train`) is produced with merged ranking values.

5. **Train the Feature Learning Model (GCN):**
   - Run:
     ```bash
     python feature_learning_gnn_gpu.py
     ```
   *Output:* A GNN model is trained and saved under the `model/` folder; training logs are written to `reports/`.

6. **Test the Model and Predict Feature Rankings:**
   - Run:
     ```bash
     python test_model.py
     ```
   *Output:* Predicted rankings and importance scores are output to `reports/rank_predicted/` and `reports/importance_predicted/`.

7. **Train the Node Classification Model:**
   - Run:
     ```bash
     python train_nc_gnn_optimized_gpu.py
     ```
   *Output:* A node classification model is trained using the predicted feature rankings; evaluation metrics are saved to `accuracy_reports/` or `reports/`.


