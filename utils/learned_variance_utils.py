import torch
from torch_geometric.utils import remove_self_loops
from utils.rotation_utils import superfibonacci_rotations


def _complete_graph(num_nodes: int, device):
    idx = torch.arange(num_nodes, device=device)
    row = idx.repeat_interleave(num_nodes)
    col = idx.repeat(num_nodes)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index


def get_orbit_variance(model, pos, z, y, k: int):
    """
    Measure equivariance for a *single molecule* under k rotations.

    model : Vanilla (or similar) GNN
    pos   : (N, 3) atom positions (for ONE molecule)
    z     : (N,) atomic numbers
    y     : (3,) dipole vector for that molecule
    k     : number of superfibonacci rotations
    """
    device = next(model.parameters()).device

    pos = pos.to(device)
    z   = z.to(device)
    y   = y.to(device)

    # 🔍 Sanity check: we expect a 3D vector (dipole)
    if y.numel() != 3:
        raise ValueError(
            f"get_orbit_variance expects y with 3 elements (dipole), "
            f"but got {y.numel()} elements. Did you pass the right target?"
        )

    y = y.view(1, 3)   # (1,3)

    N = pos.size(0)
    batch = torch.zeros(N, dtype=torch.long, device=device)
    edge_index = _complete_graph(N, device=device)

    R = superfibonacci_rotations(k, device=device, dtype=pos.dtype)  # (k,3,3)

    preds = []
    targets = []

    model.eval()
    with torch.no_grad():
        for i in range(k):
            Ri = R[i]             # (3,3)
            pos_rot = pos @ Ri.T  # (N,3)
            y_rot = (y @ Ri.T)    # (1,3)

            pred, _ = model(z=z, pos=pos_rot, edge_index=edge_index, batch=batch)
            preds.append(pred.squeeze(0))      # (3,)
            targets.append(y_rot.squeeze(0))   # (3,)

    preds = torch.stack(preds, dim=0)      # (k,3)
    targets = torch.stack(targets, dim=0)  # (k,3)

    mse_per_rotation = ((preds - targets) ** 2).mean(dim=1)  # (k,)
    orbit_variance = preds.var(dim=0, unbiased=False).mean().item()
    mean_mse = mse_per_rotation.mean().item()

    return mse_per_rotation, orbit_variance, mean_mse
