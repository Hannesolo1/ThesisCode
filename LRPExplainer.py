import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class LRPExplainer:
    """
    A minimal LRP explainer for a 2-layer GCN, inspired by
    https://github.com/baldassarreFe/graph-network-explainability

    - We store forward activations in self.activations
    - Then in attribute(...), we do the layer-wise relevance distribution.
    """

    def __init__(self, model, eps=1e-6):
        """
        Args:
            model: A trained GCN model (2-layer).
            eps:   Epsilon for numeric stability in the LRP denominator.
        """
        self.model = model
        self.eps = eps
        self.activations = {}

    def _store_activations(self, module, input, output, name):
        """
        A helper to store the forward pass inputs/outputs.
        Typically used as a PyTorch forward hook.
        """
        # We'll just store a clone of the output for LRP usage
        self.activations[name] = output.clone().detach()

    def forward(self, x, edge_index):
        """
        Forward pass that also saves intermediate activations.
        In the original GCN, we don't have named layers, so we use hooks or
        a custom forward that we manually instrument.

        We'll do a manual forward here for clarity, storing at each step:
          - x_0
          - x_1_pre, x_1_post
          - x_2 (final logits)
        """
        self.activations.clear()
        self.activations["x_0"] = x.clone().detach()

        # --- Layer 1 ---
        x1_pre = self.model.conv1(x, edge_index)
        self.activations["x_1_pre"] = x1_pre.clone().detach()
        x1_post = F.relu(x1_pre)
        self.activations["x_1_post"] = x1_post.clone().detach()

        # --- Layer 2 ---
        x2 = self.model.conv2(x1_post, edge_index)  # pre-softmax
        self.activations["x_2"] = x2.clone().detach()

        out = F.log_softmax(x2, dim=1)
        return out

    def attribute(self, x, edge_index, y):
        """
        Perform LRP to explain the predictions w.r.t. each node’s true label (y).
        Returns a feature relevance vector or matrix.

        Steps:
        1. Forward pass to store intermediate activations
        2. Build adjacency with GCN normalization
        3. Distribute relevance from final layer (R_2) -> hidden layer (R_1) -> input (R_0)
        4. Aggregate relevances across nodes as feature importances
        """
        device = x.device

        # 1) Forward pass with stored activations
        self.forward(x, edge_index)  # populates self.activations

        x_2 = self.activations["x_2"]         # shape: [num_nodes, out_channels]
        x_1_post = self.activations["x_1_post"]  # [num_nodes, hidden_dim]
        x_0 = self.activations["x_0"]         # [num_nodes, in_features]

        # 2) Build adjacency matrix (dense) with typical GCN normalization
        # Not part of the LRP algorithm yet, but needed for relevance propagation.
        # -> How does relevance flow between neighboring nodes?!
        num_nodes = x.size(0)
        adj_dense = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        # Make it symmetric if undirected:
        adj_dense = (adj_dense + adj_dense.T).clamp(max=1)

        deg = adj_dense.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        D_tilde = torch.diag(deg_inv_sqrt)
        A_norm = D_tilde @ adj_dense @ D_tilde

        # 3) Retrieve the model’s weights
        #    For a 2-layer GCN:
        w1 = self.model.conv1.lin.weight  # shape [hidden_dim, in_features]
        w2 = self.model.conv2.lin.weight  # shape [out_channels, hidden_dim]

        # --- R_2 initialization ---
        # We'll consider the sum of logit outputs for each node's true class.
        # If you want node-by-node relevance, you can do it individually in a loop.
        # shape [num_nodes, out_channels]
        # True class for each node => gather => shape [num_nodes]
        node_logits = x_2[torch.arange(num_nodes), y]
        # We'll treat that as R_2 => the relevance at the final layer for each node
        R_2 = node_logits.clone()

        # ========== LAYER 2 LRP ==========
        # R_1 => shape [num_nodes, hidden_dim]
        R_1 = torch.zeros(num_nodes, w1.size(0), device=device)

        # For each node j, gather R_2[j], pick w2[y[j], :],
        # compute contribution from neighbors i (like the Pope approach)
        for j in range(num_nodes):
            class_j = y[j].item()
            w2_class = w2[class_j, :]  # shape [hidden_dim]

            # For layer2, z_{i->j} = A_norm[i,j] * (x_1_post[i] dot w2_class)
            x1w = (x_1_post * w2_class)  # shape [num_nodes, hidden_dim]
            x1w = x1w.sum(dim=1)         # shape [num_nodes]

            z_ij = A_norm[:, j] * x1w    # shape [num_nodes]
            zsum_j = z_ij.sum() + self.eps * torch.sign(z_ij.sum())

            if zsum_j.abs() < self.eps:
                continue  # avoid division by zero

            # Distribute R_2[j] back to each node i’s hidden dimension proportionally
            ratio = z_ij / zsum_j  # shape [num_nodes]

            # Then R_1[i,:] += ratio[i] * R_2[j] * w2_class
            for i in range(num_nodes):
                R_1[i, :] += ratio[i] * R_2[j] * w2_class

        # ========== LAYER 1 LRP ==========
        # R_0 => shape [num_nodes, in_features]
        R_0 = torch.zeros(num_nodes, x_0.size(1), device=device)

        # Now we distribute R_1[j, :] back to x_0.
        # Summation approach: w1[h, in_features], R_1[j,h].
        for j in range(num_nodes):
            # sum across hidden dims to form an "effective" weight for the j-th node
            # weighting each hidden dim h by R_1[j,h].
            # w1_eff => shape [in_features]
            # w1 shape => [hidden_dim, in_features]
            # R_1[j,:] => [hidden_dim]
            w1_eff = torch.matmul(R_1[j, :], w1)  # shape [in_features]

            # For layer1, z_{i->j} = A_norm[i,j] * (x_0[i] dot w1_eff)
            x0w = (x_0 * w1_eff)  # shape [num_nodes, in_features]
            x0w = x0w.sum(dim=1)  # shape [num_nodes]
            z_ij = A_norm[:, j] * x0w  # shape [num_nodes]

            zsum_j = z_ij.sum() + self.eps * torch.sign(z_ij.sum())
            if zsum_j.abs() < self.eps:
                continue

            ratio = z_ij / zsum_j  # shape [num_nodes]
            # R_1[j] is a vector; for simplicity, we sum it all:
            Rj_total = R_1[j, :].sum()

            for i in range(num_nodes):
                R_0[i, :] += ratio[i] * Rj_total * w1_eff

        # 4) Now R_0 is node-wise, feature-wise relevance => shape [num_nodes, in_features]
        # For a single vector of feature importances across the entire subgraph,
        # we can replicate the approach of taking absolute value & average across nodes:
        feature_relevance = R_0.abs().mean(dim=0)  # shape [in_features]

        return feature_relevance.detach().cpu().numpy()
