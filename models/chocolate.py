import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from config.settings import (
    HIDDEN_DIM,
    N_LAYERS,
    EMBEDDING_SIZE,
    EPSILON,
)

#
# Equivariant Model
#

class Chocolate(nn.Module):
    def __init__(self, hidden_dim = HIDDEN_DIM, n_layers = N_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(EMBEDDING_SIZE, hidden_dim)

        self.layers = nn.ModuleList([
            Chocolate_Layer(hidden_dim=self.hidden_dim)
            for _ in range(self.n_layers)
        ])

        self.final = nn.Linear(self.hidden_dim, 1, bias=False) # Dipole prediction head

    def forward(self, z, pos, edge_index, batch):

        x = self.embedding(z)  # (N, F)
        v = pos.unsqueeze(-1).expand(-1, 3, self.hidden_dim)  # (N,3,F)

        i, j = edge_index
        r_ij = pos[j] - pos[i]
        dist_ij = r_ij.norm(dim=-1)
        


        for layer in self.layers:
            x, v = layer(x, v, edge_index, r_ij, dist_ij)

        v_flat = v.view(v.size(0), -1)  # (N, 3*F)
        mol_v_flat = global_add_pool(v_flat, batch)  # (B, 3*F)
        mol_v = mol_v_flat.view(-1, 3, self.hidden_dim)  # (B, 3, F)
        dip = self.final(mol_v).squeeze(-1)  # (B, 3)

        return dip, None

class Chocolate_Layer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message MLP: Source scalars + target scalars + distance -> gates
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.SiLU()
        )

        # Update MLP: Takes current scalar + aggregated scalar message
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim) # Output new features
        )

        # Channel-mixing linear shared across x/y/z (keeps equivariance)
        self.vector_mix = nn.Linear(hidden_dim, hidden_dim, bias=False)
        


    def forward(self, x, v, edge_index, r_ij, dist_ij):

        row, col = edge_index 

        # Scalar message
        m_in = torch.cat([x[row], x[col], dist_ij.unsqueeze(-1)], dim=-1)  # (E, 2*F + 1)
        messages = self.message_mlp(m_in)  # (E, 4*F)
        gate_vec, msg_scalar = torch.split(messages, self.hidden_dim, dim=-1)
        
        dir_ij = r_ij / dist_ij.clamp_min(EPSILON).unsqueeze(-1)  # (E, 3)
        message_vec = gate_vec.unsqueeze(1) * dir_ij.unsqueeze(-1)  # (E, 3, F)

        agg_s = torch.zeros_like(x).index_add_(0, row, msg_scalar)
        agg_v = torch.zeros_like(v).index_add_(0, row, message_vec)

        x_new = x + self.update_mlp(torch.cat([x, agg_s], dim=-1))
        v_mix = torch.einsum('nik,kh->nih', v, self.vector_mix.weight)  # mix feature channels, keep axes separate
        v_new = v + agg_v + v_mix
        return x_new, v_new
