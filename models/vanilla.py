import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from config.settings import (
    HIDDEN_DIM,
    N_LAYERS,
    EMBEDDING_SIZE,
)

class Vanilla(nn.Module):
    def __init__(self, hidden_dim = HIDDEN_DIM, n_layers = N_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(EMBEDDING_SIZE, hidden_dim)

        self.layers = nn.ModuleList([
            Vanilla_Layer(hidden_dim=self.hidden_dim)
            for _ in range(self.n_layers)
        ])

        self.final = nn.Linear(self.hidden_dim, 1) # Dipole prediction head

    def forward(self, z, pos, edge_index, batch):

        x = self.embedding(z)  # (N, F)
        v = pos.unsqueeze(-1).expand(-1, 3, self.hidden_dim)
        
        intermediates = []

        for layer in self.layers:
            x, v = layer(x, v, edge_index)
            intermediates.append((x, v))
        
        # 1. Flatten (N, 3, F) -> (N, 3*F) so global_pool sees it as "just features"
        n_nodes, _, f_dim = v.shape
        v_flat = v.view(n_nodes, -1) 
        
        # 2. Pool: Results in (Batch_Size, 3*F)
        mol_v_flat = global_add_pool(v_flat, batch)
        
        # 3. Reshape back: (Batch_Size, 3*F) -> (Batch_Size, 3, F)
        # We use -1 for the batch dim to handle variable batch sizes automatically
        mol_v = mol_v_flat.view(-1, 3, f_dim)

        # Now proceed with your prediction head
        dip = self.final(mol_v) # (B, 3, 1)
        dip = dip.squeeze(-1)   # (B, 3)

        return dip, intermediates

class Vanilla_Layer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input: Scalar features (F) + Flattened Vector features (3*F)
        node_feature_dim = hidden_dim + (3 * hidden_dim)
        
        # Message MLP: Takes Source Node + Target Node + Distance
        # Input size: 2 * node_feature_dim + 1 (distance)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_feature_dim + 1, node_feature_dim),
            nn.SiLU(),
            nn.Linear(node_feature_dim, node_feature_dim),
            nn.SiLU()
        )

        # Update MLP: Takes current Node + Aggregated Message
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_feature_dim, node_feature_dim),
            nn.SiLU(),
            nn.Linear(node_feature_dim, node_feature_dim) # Output new features
        )
        


    def forward(self, x, v, edge_index):

        row, col = edge_index 

        # 1. FLATTEN THE VECTORS
        # v shape: (N, 3, F) -> (N, 3*F)
        # This treats spatial geometry as just "more features"
        v_flat = v.reshape(v.size(0), -1) 
        
        # Combine scalars and vectors into one big feature vector
        # node_feats shape: (N, 4*F)
        node_feats = torch.cat([x, v_flat], dim=-1)

        # 2. CALCULATE GEOMETRY (Just for edge attributes)
        # We still likely want distance, or even relative vectors, as input features
        pseudo_pos = v.mean(dim=-1) # (N, 3)
        rij = pseudo_pos[row] - pseudo_pos[col]
        dist = rij.norm(dim=-1, keepdim=True)

        # 3. MESSAGE PASSING (Standard MPNN style)
        # Concatenate Source + Target + Edge Attribute
        edge_input = torch.cat([node_feats[row], node_feats[col], dist], dim=-1)
        
        # Learn arbitrary relationships between x, y, z, and scalars
        messages = self.message_mlp(edge_input) # (E, 4*F)

        # Aggregate messages
        aggregated = torch.zeros_like(node_feats)
        aggregated.index_add_(0, row, messages)

        # 4. UPDATE NODE FEATURES
        # Combine old features with aggregated messages
        update_input = torch.cat([node_feats, aggregated], dim=-1)
        new_node_feats = self.update_mlp(update_input) # (N, 4*F)

        # 5. UNFLATTEN / RESHAPE
        # Split back into scalar (N, F) and vector (N, 3*F)
        x_new = new_node_feats[:, :self.hidden_dim]
        v_new_flat = new_node_feats[:, self.hidden_dim:]
        
        # Reshape vector part back to (N, 3, F)
        v_new = v_new_flat.view(v.size(0), 3, self.hidden_dim)

        return x_new, v_new