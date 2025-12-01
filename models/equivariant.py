import torch
import torch.nn as nn

from config.settings import (
    HIDDEN_DIM,
    N_LAYERS,
    N_RBF,
    CUTOFF,
    DEVICE,
    EPSILON,
    EMBEDDING_SIZE,
)

#ADAPTED FROM https://github.com/nityasagarjena/PaiNN-model

def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    idx = torch.arange(edge_size, device=edge_dist.device, dtype=edge_dist.dtype) + 1
    dist = edge_dist.unsqueeze(-1).clamp_min(EPSILON)
    return torch.sin(dist * idx * torch.pi / cutoff) / dist


def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    cutoff_tensor = torch.zeros_like(edge_dist)
    mask = edge_dist < cutoff
    cutoff_tensor[mask] = 0.5 * (torch.cos(torch.pi * edge_dist[mask] / cutoff) + 1.0)
    return cutoff_tensor


class PainnMessage(nn.Module):
    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()
        self.edge_size = edge_size
        self.node_size = node_size
        self.cutoff = cutoff

        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )

        self.filter_layer = nn.Linear(edge_size, node_size * 3)

    def forward(self, node_scalar, node_vector, edge_index, edge_diff, edge_dist):
        filter_weight = self.filter_layer(sinc_expansion(edge_dist, self.edge_size, self.cutoff))
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)
        scalar_out = self.scalar_message_mlp(node_scalar)
        scalar_j = scalar_out[edge_index[1]]
        filter_out = filter_weight * scalar_j

        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out,
            self.node_size,
            dim=1,
        )

        message_vector = node_vector[edge_index[1]] * gate_state_vector.unsqueeze(1)
        edge_dir = edge_diff / edge_dist.clamp_min(EPSILON).unsqueeze(-1)
        edge_vector = gate_edge_vector.unsqueeze(1) * edge_dir.unsqueeze(-1)
        message_vector = message_vector + edge_vector

        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)
        residual_scalar.index_add_(0, edge_index[0], message_scalar)
        residual_vector.index_add_(0, edge_index[0], message_vector)

        return node_scalar + residual_scalar, node_vector + residual_vector


class PainnUpdate(nn.Module):
    def __init__(self, node_size: int):
        super().__init__()
        self.update_U = nn.Linear(node_size, node_size, bias=False)
        self.update_V = nn.Linear(node_size, node_size, bias=False)

        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )

    def _apply_linear(self, tensor, linear):
        N = tensor.shape[0]
        out = linear(tensor.view(N * 3, -1))
        return out.view(N, 3, -1)

    def forward(self, node_scalar, node_vector):
        Uv = self._apply_linear(node_vector, self.update_U)
        Vv = self._apply_linear(node_vector, self.update_V)

        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)

        a_vv, a_sv, a_ss = torch.split(
            mlp_output,
            node_vector.shape[-1],
            dim=1,
        )

        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss

        return node_scalar + delta_s, node_vector + delta_v


class EquivariantModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        n_rbf: int = N_RBF,
        cutoff: float = CUTOFF,
        device: torch.device = DEVICE,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.device = device

        self.embedding = nn.Embedding(EMBEDDING_SIZE, hidden_dim)

        self.message_layers = nn.ModuleList(
            [PainnMessage(hidden_dim, n_rbf, cutoff) for _ in range(n_layers)]
        )

        self.update_layers = nn.ModuleList(
            [PainnUpdate(hidden_dim) for _ in range(n_layers)]
        )

        self.readout = nn.Linear(hidden_dim, 3)

    def forward(self, z, pos, edge_index, batch):
        s = self.embedding(z)
        v = s.new_zeros((s.size(0), 3, self.hidden_dim))

        i, j = edge_index
        edge_diff = pos[j] - pos[i]
        edge_dist = torch.linalg.norm(edge_diff, dim=1)

        for msg_layer, upd_layer in zip(self.message_layers, self.update_layers):
            s, v = msg_layer(s, v, edge_index, edge_diff, edge_dist)
            s, v = upd_layer(s, v)

        per_atom_vec = self.readout(s)
        B = int(batch.to(per_atom_vec.device).max().item()) + 1
        pred = per_atom_vec.new_zeros((B, per_atom_vec.size(1)))
        pred.index_add_(0, batch, per_atom_vec)
        return pred
