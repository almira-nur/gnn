import math
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# recommended constants from the paper
phi = math.sqrt(2)
psi = 1.533751168755204288118041  


def superfibonacci_quaternions(num_samples, device=None, dtype=torch.float32):
    """
    Generate `num_samples` unit quaternions distributed quasi-uniformly on SO(3)
    using the Super-Fibonacci method.
    Returns quaternions in (x, y, z, w) format.
    """
    if device is None:
        device = torch.device("cpu")

    i = torch.arange(num_samples, device=device, dtype=dtype)
    s = i + 0.5
    t = s / num_samples
    r = torch.sqrt(t)
    R = torch.sqrt(1 - t)

    alpha = 2 * math.pi * s / phi
    beta  = 2 * math.pi * s / psi

    qx = r * torch.sin(alpha)
    qy = r * torch.cos(alpha)
    qz = R * torch.sin(beta)
    qw = R * torch.cos(beta)

    q = torch.stack([qx, qy, qz, qw], dim=-1)  # (n,4)
    q = q / q.norm(dim=-1, keepdim=True)
    return q


def quaternion_to_matrix(q):

    #Convert quaternion(s) (x,y,z,w) to rotation matrix/matrices.

    x, y, z, w = q.unbind(-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    R = torch.stack([
        1 - 2*(yy + zz),   2*(xy - zw),       2*(xz + yw),
        2*(xy + zw),       1 - 2*(xx + zz),   2*(yz - xw),
        2*(xz - yw),       2*(yz + xw),       1 - 2*(xx + yy)
    ], dim=-1)

    return R.reshape(q.shape[:-1] + (3, 3))


def superfibonacci_rotations(num_samples, device=None, dtype=torch.float32):
    """Generate num_samples SO(3) rotation matrices in a batch"""
    q = superfibonacci_quaternions(num_samples, device=device, dtype=dtype)
    return quaternion_to_matrix(q)


def rotate_positions(pos, R):
    """
    pos: (N_atoms, 3)
    R:   (n_rot, 3, 3)
    returns: (n_rot, N_atoms, 3)
    """
    pos_expanded = pos.unsqueeze(0).expand(R.shape[0], -1, -1)
    return pos_expanded @ R.transpose(1, 2)



def superfib_augment_batch(pos, z, y, batch, k, device=None):
    
    if device is None:
        device = pos.device

    
    B = int(batch.max().item()) + 1
    N_atoms = pos.size(0)

    # Center each molecule
    pos_centered = pos.clone()
    for mol_idx in range(B):
        mask = (batch == mol_idx)
        center_of_mass = pos[mask].mean(dim=0)
        pos_centered[mask] = pos[mask] - center_of_mass

    # Generate k rotations
    
    R = superfibonacci_rotations(k, device=device, dtype=pos_centered.dtype)  # (k, 3, 3)

    # Apply rotations
    pos_rot = rotate_positions(pos_centered, R)         # (k, N_atoms, 3)
    pos_out = pos_rot.reshape(k * N_atoms, 3)  # (k*N_atoms, 3)

    # Compute new batch indices

    batch = batch.to(device)
    batch_tiled = batch.unsqueeze(0).expand(k, -1)  # (k, N_atoms)

    offsets = (torch.arange(k, device=device) * B).view(k, 1)  # (k,1)
    batch_aug = batch_tiled + offsets                          # (k, N_atoms)
    batch_out = batch_aug.reshape(k * N_atoms)

    # Expand Z

    z = z.to(device)
    if z.dim() == 1:
        # (N_atoms,) -> (k*N_atoms,)
        z_out = z.unsqueeze(0).expand(k, -1).reshape(-1)
    else:
        # (N_atoms,f) -> (k*N_atoms,f)
        Fz = z.size(-1)
        z_out = (
            z.unsqueeze(0)
             .expand(k, -1, -1)
             .reshape(k * N_atoms, Fz)
        )

    # Expand y

    y = y.to(device)
    y_was_1d = (y.dim() == 1)
    if y_was_1d:
        y = y.unsqueeze(-1)  # (B,1)

    y_out = (
        y.unsqueeze(0)               # (1,B,1)
          .expand(k, -1, -1)         # (k,B,1)
          .reshape(k * B, -1)        # (k*B,1)
    )

    if y_was_1d:
        y_out = y_out.view(-1)       # (k*B,)

    return pos_out, z_out, y_out, batch_out


def pairwise_distances(x):
    """
    x: (N_atoms, 3)
    returns: (N_atoms, N_atoms) distance matrix
    """
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # (N,N,3)
    return torch.linalg.norm(diff, dim=-1)


def verify_rotation_preserves_distances(pos, pos_rot, atol=1e-6):
    """
    pos:     original (N,3)
    pos_rot: rotated  (N,3)
    """
    D0 = pairwise_distances(pos)
    D1 = pairwise_distances(pos_rot)

    diff = (D0 - D1).abs()
    max_err = diff.max().item()
    print("Max pairwise distance error:", max_err)

    return max_err < atol
