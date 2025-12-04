import torch

checkpoint = torch.load("/Users/User/Downloads/project/gnn/checkpoints/checkpoint_epoch_1.pt")

print(type(checkpoint))
print(checkpoint.keys() if isinstance(checkpoint, dict) else checkpoint)

import os
import torch
import matplotlib.pyplot as plt

from config.settings import CHECKPOINT_PATH, FIG_PATH
device = torch.device("cpu")

checkpoints = []
for fname in os.listdir(CHECKPOINT_PATH):
    if fname.startswith("checkpoint_epoch_") and fname.endswith(".pt"):
        path = os.path.join(CHECKPOINT_PATH, fname)
        ckpt = torch.load(path, map_location="cpu")
        checkpoints.append(ckpt)

# sort by epoch just in case
checkpoints.sort(key=lambda c: c["epoch"])

epochs = [c["epoch"] for c in checkpoints]
val_losses = [c["val_loss"] for c in checkpoints]
rot_losses = [c["rotate_val_loss"] for c in checkpoints]

gaps = [rv - v for v, rv in zip(val_losses, rot_losses)]
rel_gaps = [(rv / v - 1.0) if v > 0 else 0.0 for v, rv in zip(val_losses, rot_losses)]

print("epoch | val_loss | rot_val_loss | gap | rel_gap")
for e, v, rv, g, rg in zip(epochs, val_losses, rot_losses, gaps, rel_gaps):
    print(f"{e:5d} | {v:.6f} | {rv:.6f} | {g:.6f} | {rg*100:6.2f}%")

# Plot relative gap over epochs
plt.figure(figsize=(6, 4))
plt.plot(epochs, rel_gaps, marker="o")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Relative equivariance gap (rot_val / val - 1)")
plt.title("Equivariance gap vs epoch")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "equivariance_gap_over_epochs.png"))
plt.close()
import os
import torch
from torch_geometric.loader import DataLoader

from models.vanilla import Vanilla
from qm7x_dataset import QM7XDataset
from utils.learned_variance_utils import get_orbit_variance
from config.settings import HIDDEN_DIM, N_LAYERS, VAL_DATA, CHECKPOINT_PATH

device = torch.device("cpu")  # or cuda if available

# 1. Build a tiny val loader with batch_size=1 for simplicity
val_dataset = QM7XDataset(VAL_DATA)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
probe_batch = next(iter(val_loader))
probe_batch = probe_batch.to(device)

pos_i = probe_batch.pos      # (N,3)
z_i   = probe_batch.z        # (N,)
dip_i = probe_batch.dip      # should contain dipole

print("pos_i shape:", pos_i.shape)
print("z_i shape:", z_i.shape)
print("raw dip_i shape:", dip_i.shape)

y_i = dip_i.view(-1)         # flatten to 1D
print("flattened y_i shape:", y_i.shape, "numel:", y_i.numel())

# If this prints numel=3, you’re good.
# If it prints numel=1, your dataset is not giving dipoles the way you expect.

k = 20  # number of rotations to probe

# 2. Loop over checkpoints and compute orbit variance
for fname in sorted(os.listdir(CHECKPOINT_PATH)):
    if not (fname.startswith("checkpoint_epoch_") and fname.endswith(".pt")):
        continue

    path = os.path.join(CHECKPOINT_PATH, fname)
    ckpt = torch.load(path, map_location=device)
    epoch = ckpt["epoch"]

    model = Vanilla(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mse_rot, orbit_var, mean_mse = get_orbit_variance(model, pos_i, z_i, y_i, k)

    print(
        f"Epoch {epoch:3d}: orbit_variance={orbit_var:.6e}, "
        f"mean_mse={mean_mse:.6e}"
    )

