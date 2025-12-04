# egnn_layers.py
import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean

# -------------------
# EGNN Layer
# -------------------

class EGNNLayer(nn.Module):
    def __init__(self, in_dim, edge_dim=0, hidden_dim=64):
        super().__init__()
        edge_in = 2 * in_dim + 1 + edge_dim  # hi, hj, dist2, (edge_attr)

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # keeps coordinate steps bounded
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        h: [N, F] node features
        x: [N, 3] coordinates
        edge_index: [2, E] (row, col)
        edge_attr: [E, edge_dim] or None
        """
        row, col = edge_index  # messages from col -> row

        rel = x[row] - x[col]                    # [E, 3]
        dist2 = (rel ** 2).sum(-1, keepdim=True) # [E, 1]

        if edge_attr is None:
            edge_input = torch.cat([h[row], h[col], dist2], dim=-1)
        else:
            edge_input = torch.cat([h[row], h[col], dist2, edge_attr], dim=-1)

        m_ij = self.edge_mlp(edge_input)         # [E, H]

        # coordinate update (equivariant)
        coef = self.coord_mlp(m_ij)              # [E, 1]
        trans = rel * coef                       # [E, 3]
        dx = scatter_sum(trans, row, dim=0, dim_size=h.size(0))  # [N, 3]
        x = x + dx

        # feature update
        m_agg = scatter_sum(m_ij, row, dim=0, dim_size=h.size(0))  # [N, H]
        node_in = torch.cat([h, m_agg], dim=-1)
        dh = self.node_mlp(node_in)
        h = h + dh  # residual

        return h, x


# -------------------
# EGNN Dipole Model
# -------------------

class EGNNDipoleModel(nn.Module):
    """
    EGNN that predicts a dipole vector per molecule.
    Output is SE(3)-equivariant:
      x -> R x  ==>  mu -> R mu
    and translation invariant.
    """

    def __init__(
        self,
        num_atom_types=100,
        in_dim=64,
        hidden_dim=64,
        n_layers=4,
        edge_dim=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, in_dim)

        self.layers = nn.ModuleList(
            [EGNNLayer(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
             for _ in range(n_layers)]
        )

        # scalar "charge" per node from features
        self.charge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, pos, edge_index, batch, edge_attr=None):
        """
        z: [N] atomic numbers (int64)
        pos: [N, 3]
        edge_index: [2, E]
        batch: [N] graph index per node
        """
        h = self.embedding(z)
        x = pos

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr=edge_attr)

        # per-node scalar "charges"
        q = self.charge_mlp(h).squeeze(-1)   # [N]

        # per-graph center (mean position)
        x_center = scatter_mean(x, batch, dim=0)  # [G, 3]
        x_rel = x - x_center[batch]              # [N, 3]

        # dipole contributions and graph sum
        dip_node = q.unsqueeze(-1) * x_rel       # [N, 3]
        mu = scatter_sum(dip_node, batch, dim=0) # [G, 3]

        return mu

    def encode(self, z, pos, edge_index, batch, edge_attr=None, return_coords=False):
        """
        Return graph-level latent representations (mean of node features).
        """
        h = self.embedding(z)
        x = pos
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr=edge_attr)
        # graph-level embedding
        g = scatter_mean(h, batch, dim=0)  # [G, F]
        if return_coords:
            return g, x
        return g

