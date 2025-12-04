import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
# Assuming config settings are available
from config.settings import (
    HIDDEN_DIM,
    N_LAYERS,
    EMBEDDING_SIZE,
)

class Strawberry(nn.Module):
    def __init__(self, hidden_dim = HIDDEN_DIM, n_layers = N_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(EMBEDDING_SIZE, hidden_dim)

        self.layers = nn.ModuleList([
            Strawberry_Layer(hidden_dim=self.hidden_dim)
            for _ in range(self.n_layers)
        ])

        self.final = nn.Linear(self.hidden_dim, 1, bias=False) # Dipole prediction head

    def forward(self, z, pos, edge_index, batch):

        x = self.embedding(z)  # (N, F)
        v = pos.unsqueeze(-1).expand(-1, 3, self.hidden_dim)
        
        # Intermediates storage can be kept or removed based on need
        intermediates = [] 

        for layer in self.layers:
            x, v = layer(x, v, edge_index)
            intermediates.append((x, v))
        
        # 1. Flatten (N, 3, F) -> (N, 3*F)
        n_nodes, _, f_dim = v.shape
        v_flat = v.view(n_nodes, -1) 
        
        # 2. Pool: Results in (Batch_Size, 3*F)
        mol_v_flat = global_add_pool(v_flat, batch)
        
        # 3. Reshape back: (Batch_Size, 3*F) -> (Batch_Size, 3, F)
        mol_v = mol_v_flat.view(-1, 3, f_dim)

        # Prediction head
        dip = self.final(mol_v) # (B, 3, 1)
        dip = dip.squeeze(-1)   # (B, 3)

        return dip, intermediates

class Strawberry_Layer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        node_feature_dim = hidden_dim + (3 * hidden_dim) # 4*F

        # Message MLP: Takes Source Node + Target Node + Distance
        # Input size: 2 * node_feature_dim + 1 = 8*F + 1
        # Output size: node_feature_dim = 4*F (will be split into m_s (F) and m_v_flat (3*F))
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_feature_dim + 1, node_feature_dim),
            nn.SiLU(),
            nn.Linear(node_feature_dim, node_feature_dim),
            nn.SiLU()
        )

        # SCALAR Update MLP: Takes current scalar (F) + aggregated scalar (F) -> 2*F input
        # Output new scalar features (F)
        self.scalar_update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), 
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim) 
        )
        
        # Channel-mixing linear shared across x/y/z (keeps the structural bias of Chocolate)
        self.vector_mix = nn.Linear(hidden_dim, hidden_dim, bias=False)
        


    def forward(self, x, v, edge_index):

        row, col = edge_index 

        # 1. Feature Preparation (Fusion for message input, Separation for aggregation targets)
        v_flat = v.reshape(v.size(0), -1) 
        # Combined features for MLP input (N, 4*F)
        node_feats = torch.cat([x, v_flat], dim=-1) 

        # 2. Geometry Calculation
        # Use pseudo-position for distance calculation (same as Vanilla/Original Strawberry)
        pseudo_pos = v.mean(dim=-1) 
        rij = pseudo_pos[row] - pseudo_pos[col]
        dist = rij.norm(dim=-1, keepdim=True)

        # 3. MESSAGE PASSING (Non-Equivariant Message Generation)
        # Input to MLP: Source(4*F) + Target(4*F) + Distance(1)
        edge_input = torch.cat([node_feats[row], node_feats[col], dist], dim=-1)
        
        # Output: Single, non-equivariant message block (E, 4*F)
        messages = self.message_mlp(edge_input) 

        # SPLIT MESSAGE COMPONENTS (Feature Separation)
        # The first F are scalar, the remaining 3*F are vector-related
        msg_scalar = messages[:, :self.hidden_dim]       # (E, F)
        msg_v_flat = messages[:, self.hidden_dim:]       # (E, 3*F)
        
        # Reshape the vector message for aggregation on v
        msg_vector = msg_v_flat.view(-1, 3, self.hidden_dim) # (E, 3, F)

        # 4. AGGREGATE MESSAGES (Separate Aggregation)
        agg_s = torch.zeros_like(x).index_add_(0, row, msg_scalar) # (N, F)
        agg_v = torch.zeros_like(v).index_add_(0, row, msg_vector) # (N, 3, F)

        # 5. UPDATE NODE FEATURES (Separate Update, with Chocolate's structural terms)
        
        # Update scalar features (x)
        x_update_input = torch.cat([x, agg_s], dim=-1)
        x_new = x + self.scalar_update_mlp(x_update_input)
        
        # Calculate v_mix (maintains structural term from Chocolate)
        v_mix = torch.einsum('nik,kh->nih', v, self.vector_mix.weight)
        
        # Update vector features (v)
        v_new = v + agg_v + v_mix # Includes non-equiv. message (agg_v) and mix term (v_mix)

        return x_new, v_new