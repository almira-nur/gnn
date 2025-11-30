import torch
import torch.nn as nn
from torch_scatter import scatter_add

from config.settings import HIDDEN_DIM, N_LAYERS, N_RBF, CUTOFF, DEVICE, EPSILON, EMBEDDING_SIZE

class EquivariantModel(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, n_rbf=N_RBF, cutoff=CUTOFF, device=DEVICE):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.device = device

        #Well, let's check if we need to do an embedding.
        #Maximum atomic number is for Cl (17)
        self.embedding = nn.Embedding(EMBEDDING_SIZE, hidden_dim)

        self.layers = nn.ModuleList(
            [PaiNNLayer(hidden_dim=hidden_dim, n_rbf=n_rbf, cutoff=cutoff) for _ in range(n_layers)]
        )

        self.readout = nn.Linear(hidden_dim, 3)

    def forward(self, z, pos, edge_index, batch):
        s = self.embedding(z)
        v = s.new_zeros((s.size(0), 3, self.hidden_dim))
        for layer in self.layers:
            s, v = layer(s, v, pos, edge_index)
        
        per_atom_vec = self.readout(s)  # (N, 3)
        pred = scatter_add(per_atom_vec, batch, dim=0)  # (B, 3)
        return pred
        
        
    
class PaiNNLayer(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, n_rbf=N_RBF, cutoff=CUTOFF):
        super().__init__()
        self.cutoff = cutoff
        self.hidden_dim = hidden_dim
        self.n_rbf = n_rbf
        self.rbf = RBF(n_rbf=self.n_rbf, cutoff=self.cutoff)

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

        #For numerical stability
        dist_safe = dist.clamp_min(EPSILON)

        unit = rij / (dist_safe.unsqueeze(-1))

        w_ij = self.phi(self.rbf(dist_safe))  # (E, F)

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

        N = m_v.shape[0]
        m_v_flat = m_v.view(N * 3, self.hidden_dim)
        m_v = self.U_v(m_v_flat).view(N, 3, self.hidden_dim)
        v = v + m_v

        return s, v

f

class RBF(nn.Module):

    def __init__(self, n_rbf=N_RBF, cutoff=CUTOFF):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff

        self.register_buffer("freqs", torch.arange(1, n_rbf+1, dtype=torch.float32))

    def forward(self, r):

        r = r.clamp_min(EPSILON)

        denom = float(self.cutoff) if (self.cutoff is not None) else 1.0

        x = (r.unsqueeze(-1) * self.freqs * torch.pi) / denom
        rbf = torch.sin(x) / r.unsqueeze(-1)
        
        #Cutoff block.  Maybe worth removing.
        if self.cutoff is not None:
            cutoff_mask = (r < self.cutoff).float()
            cutoff_val = 0.5 * (torch.cos(torch.pi * r / self.cutoff) + 1.0)
            cutoff_val = cutoff_val * cutoff_mask  # zero outside cutoff

            # Multiply each RBF channel by cutoff (broadcast to (E, n_rbf))
            rbf = rbf * cutoff_val.unsqueeze(-1)

        return rbf

