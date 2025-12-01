import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from qm7x_dataset import QM7XDataset
from models.equivariant import EquivariantModel
from tqdm import tqdm
from config.settings import DEVICE, SEED, BATCH_SIZE, LR, HIDDEN_DIM, N_LAYERS, NUM_EPOCHS, SHUFFLE, CHECKPOINT_PATH

import glob
import matplotlib.pyplot as plt

ckpt_paths = sorted(glob.glob(f"{CHECKPOINT_PATH}/checkpoint_epoch_*.pt"),
                    key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))

epochs, train_losses, val_losses = [], [], []
for path in ckpt_paths:
    ckpt = torch.load(path, map_location="cpu")
    epochs.append(ckpt["epoch"])
    train_losses.append(ckpt["loss"])
    val_losses.append(ckpt["val_loss"])

plt.figure(figsize=(8,4))
plt.plot(epochs, train_losses, marker="o", label="Train")
plt.plot(epochs, val_losses, marker="o", label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve from Checkpoints")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve_from_checkpoints.png")
plt.close()
print("Saved loss_plot to loss_curve_from_checkpoints.png")