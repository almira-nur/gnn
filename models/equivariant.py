import torch
import torch.nn as nn
from torch_scatter import scatter_add

from config.settings import *

class EquivariantModel:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class PaiNNLayer(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.weight = weight

        self.rbf = RBF()

        self.phi = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        #Scalar update
        self.U_s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        #Vector update
        self.U_v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, s, v, pos, edge_index):
        i, j = edge_index
        rij = pos[j] - pos[i]         # (E, 3)
        dist = torch.norm(rij, dim=-1)
        unit = rij / (dist.unsqueeze(-1) + 1e-9)

        w_ij = self.phi(self.rbf(dist))  # (E, F)

        s_j = s[j]                    # (E, F)
        v_j = v[j]                    # (E, 3, F)

        # Scalar message
        m_s_ij = self.W_s(s_j) * w_ij

        # Project v_j onto unit directions (v_j · r_hat_ij)
        proj = (v_j * unit.unsqueeze(-1)).sum(dim=1)  # (E, F)

        # Vector message
        m_v_ij = self.W_v(proj) * w_ij                 # (E, F)
        m_v_ij = unit.unsqueeze(-1) * m_v_ij.unsqueeze(-2)  # (E, 3, F)

        # Aggregate to nodes
        m_s = scatter_add(m_s_ij, i, dim=0, dim_size=s.size(0))
        m_v = scatter_add(m_v_ij, i, dim=0, dim_size=v.size(0))

        # Update
        s = s + self.U_s(m_s)
        v = v + m_v


    def __call__(self, x):
        return x * self.weight




class RBF(nn.Module):
    def __init__(self, n_rbf=N_RBF, cutoff=CUTOFF):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff

        self.register_buffer("n", torch.arange(1, n_rbf + 1).float())

    def forward(self, r):

        x = (r.unsqueeze(-1) * self.n * torch.pi) / self.cutoff
        rbf = torch.sin(x) / r.unsqueeze(-1)
        
        #Cutoff block.  Maybe worth removing.
        if self.cutoff is not None:
            cutoff_mask = (r < self.cutoff).float()
            cutoff_val = 0.5 * (torch.cos(torch.pi * r / self.cutoff) + 1.0)
            cutoff_val = cutoff_val * cutoff_mask  # zero outside cutoff

            # Multiply each RBF channel by cutoff (broadcast to (E, n_rbf))
            rbf = rbf * cutoff_val.unsqueeze(-1)

        return rbf

