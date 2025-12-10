import glob
import math
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
import numpy as np
from tqdm import tqdm
import seaborn as sns
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops

from models.chocolate import Chocolate
from models.strawberry import Strawberry
from qm7x_dataset import QM7XDataset

sns.set_theme(style="whitegrid", context="talk")
rcParams.update({
    "axes.edgecolor": "#C0C0C0",
    "axes.linewidth": 1.0,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.8,
    "axes.titleweight": "bold",
})

MODEL_TYPES = ["chocolate", "strawberry"]
AUGMENT_TYPES = ["none", "superfib_end", "superfib_intermediate"]
HIDDEN_DIM = 64
N_LAYERS = 3
BATCH_SIZE = 64
LR = 0.001
PLOT_LOGGED = True
PLOT_RECOMPUTED = False
LOG_Y_SCALE = False
SKIP_EPOCHS = {"chocolate": 1, "strawberry": 4}
LINE_WIDTH = 2.2
MARKER_SIZE = 7
MARKER_EDGE_WIDTH = 0.8
PALETTE = sns.color_palette("Set2", 2)
PRED_SCATTER_ORDER = [
    ("strawberry", "none", "Vanilla (no aug)"),
    ("strawberry", "superfib_end", "Strawberry + Augmentation"),
    ("strawberry", "superfib_intermediate", "Strawberry + Aug + Layerwise Loss"),
    ("chocolate", "none", "Chocolate (Equivariant)"),
]

DATASET_TAG = "mini_200_conf_qm7x_processed_train"
CHECKPOINT_ROOT = Path("checkpoints")
FIG_ROOT = Path("figures")
FIG_ROOT.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Candidate dataset locations
LOCAL_TRAIN = "data/mini_200_conf_qm7x_processed_train.h5"
LOCAL_VAL = "data/mini_200_conf_qm7x_processed_val.h5"
REMOTE_TRAIN = "/home/ptim/orcd/scratch/data/mini_200_conf_qm7x_processed_train.h5"
REMOTE_VAL = "/home/ptim/orcd/scratch/data/mini_200_conf_qm7x_processed_val.h5"


def pick_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return str(p)
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")


def complete_graph(num_nodes: int, device: torch.device):
    idx = torch.arange(num_nodes, device=device)
    row = idx.repeat_interleave(num_nodes)
    col = idx.repeat(num_nodes)
    return torch.stack([row, col], dim=0)


def build_block_complete_graph(batch_indices):
    device = batch_indices.device
    edge_chunks = []

    for mol_id in batch_indices.unique(sorted=True):
        node_idx = torch.nonzero(batch_indices == mol_id, as_tuple=False).view(-1)
        if node_idx.numel() == 0:
            continue

        local_edges = complete_graph(node_idx.numel(), device=device)
        edge_chunks.append(node_idx[local_edges])

    if edge_chunks:
        return torch.cat(edge_chunks, dim=1)

    return torch.empty((2, 0), dtype=torch.long, device=device)


def build_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "chocolate":
        return Chocolate(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    if model_type == "strawberry":
        return Strawberry(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    raise ValueError(f"Unknown model type: {model_type}")


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return ckpts[-1] if ckpts else None


def collect_pred_true(model, loader):
    preds, trues = [], []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, leave=False):
            batch = batch.to(DEVICE)
            z = batch.z
            pos = batch.pos
            dip = batch.dip
            b = batch.batch
            B = b.max().item() + 1
            dip = dip.view(B, 3)

            edge_index = build_block_complete_graph(b)
            edge_index, _ = remove_self_loops(edge_index)

            pred, _ = model(z=z, pos=pos, edge_index=edge_index, batch=b)
            preds.append(pred.detach().cpu())
            trues.append(dip.detach().cpu())
    model.train()
    if not preds:
        return None, None, None
    pred_arr = torch.cat(preds, dim=0).numpy().reshape(-1)
    true_arr = torch.cat(trues, dim=0).numpy().reshape(-1)
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    return true_arr, pred_arr, rmse


def plot_pred_true_grid(entries, outfile_base: str = "dipole_pred_vs_true_grid"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), squeeze=False)
    for ax, entry in zip(axes.flatten(), entries):
        if entry is None:
            ax.axis("off")
            continue
        label, true_arr, pred_arr, rmse = entry
        sns.scatterplot(
            x=true_arr,
            y=pred_arr,
            ax=ax,
            s=18,
            alpha=0.35,
            edgecolor="none",
            color=PALETTE[0],
        )
        lo = min(true_arr.min(), pred_arr.min())
        hi = max(true_arr.max(), pred_arr.max())
        pad = 0.05 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#444", linestyle="--", linewidth=1.3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("True dipole component")
        ax.set_ylabel("Predicted dipole component")
        wrapped_title = textwrap.fill(label, width=26)
        ax.set_title(wrapped_title, pad=8)
        ax.text(
            0.05,
            0.92,
            f"RMSE = {rmse:.3f}",
            transform=ax.transAxes,
            fontsize="medium",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB"),
        )
    for ax in axes.flatten()[len(entries):]:
        ax.axis("off")
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.42, wspace=0.35)
    fig.savefig(FIG_ROOT / f"{outfile_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_ROOT / f"{outfile_base}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prediction scatter grid to {FIG_ROOT / (outfile_base + '.png')} and .svg")

def evaluate_loader(loader, model):
    mse = nn.MSELoss()
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in tqdm(loader):
            batch = batch.to(DEVICE)
            z = batch.z
            pos = batch.pos
            dip = batch.dip
            b = batch.batch
            B = b.max().item() + 1
            dip = dip.view(B, 3)

            edge_index = build_block_complete_graph(b)
            edge_index, _ = remove_self_loops(edge_index)

            pred, _ = model(z=z, pos=pos, edge_index=edge_index, batch=b)
            loss = mse(pred, dip)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def checkpoint_dir(model_type: str, augment_type: str) -> Path:
    name = f"{DATASET_TAG}_{model_type}_{augment_type}_hd{HIDDEN_DIM}_nl{N_LAYERS}_bs{BATCH_SIZE}_lr{LR}"
    return CHECKPOINT_ROOT / name / "checkpoints"


def load_losses(ckpt_dir: Path, skip_epochs: int = 0):
    ckpt_paths = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]))
    if not ckpt_paths:
        return None
    epochs, train_losses, val_losses = [], [], []
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location="cpu")
        if not {"epoch", "loss", "val_loss"} <= ckpt.keys():
            raise KeyError(f"Checkpoint {path} missing required keys; found {list(ckpt.keys())}")
        epochs.append(ckpt["epoch"])
        train_losses.append(ckpt["loss"])
        val_losses.append(ckpt["val_loss"])
    filtered = [(e, t, v) for e, t, v in zip(epochs, train_losses, val_losses) if e > skip_epochs]
    if not filtered:
        return None
    epochs_f, train_f, val_f = zip(*filtered)
    return list(epochs_f), list(train_f), list(val_f)


def recompute_losses(ckpt_dir: Path, model_type: str, train_loader, val_loader, skip_epochs: int = 0):
    ckpt_paths = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]))
    if not ckpt_paths:
        return None
    recomputed = []
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location=DEVICE)
        if "model_state_dict" not in ckpt:
            print(f"Skipping recompute for {path}: missing model_state_dict")
            continue
        model = build_model(model_type).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt["epoch"]
        print(f"Recomputing Losses for {model_type}")
        train_loss = evaluate_loader(train_loader, model)
        val_loss = evaluate_loader(val_loader, model)
        recomputed.append((epoch, train_loss, val_loss))
        print(f"[Recomputed] {model_type} epoch {epoch:3d} | train={train_loss:.6f} | val={val_loss:.6f}")
    if not recomputed:
        return None
    recomputed.sort(key=lambda x: x[0])
    filtered = [(e, t, v) for e, t, v in recomputed if e > skip_epochs]
    if not filtered:
        return None
    epochs = [e for e, _, _ in filtered]
    train_losses = [t for _, t, _ in filtered]
    val_losses = [v for _, _, v in filtered]
    return epochs, train_losses, val_losses


def plot_single(epochs, train_losses, val_losses, title, outfile_base):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.lineplot(
        x=epochs,
        y=train_losses,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH,
        linewidth=LINE_WIDTH,
        label="Train",
        ax=ax,
        color=PALETTE[0],
        alpha=0.9,
    )
    sns.lineplot(
        x=epochs,
        y=val_losses,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH,
        linewidth=LINE_WIDTH,
        label="Val",
        ax=ax,
        color=PALETTE[1],
        alpha=0.9,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if LOG_Y_SCALE:
        ax.set_yscale("log")
    ax.set_title(textwrap.fill(title, width=28))
    ax.legend(frameon=True)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG_ROOT / f"{outfile_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_ROOT / f"{outfile_base}.svg", bbox_inches="tight")
    plt.close(fig)


def augment_title(augment_type: str) -> str:
    match augment_type:
        case "none":
            return "No Augmentation"
        case "superfib_end":
            return "Augmentation"
        case "superfib_intermediate":
            return "Augmentation and Layerwise Loss"
        case _:
            return augment_type


def plot_grid(curves_dict, grid_base: str, title_suffix: str = ""):
    entries = []
    for m in MODEL_TYPES:
        for a in AUGMENT_TYPES:
            curves = curves_dict.get((m, a))
            if curves is not None:
                entries.append((m, a, curves))

    if not entries:
        print(f"No data for {grid_base}; skipping grid plot.")
        return

    n = len(entries)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.4 * nrows),
        squeeze=False,
    )

    # Let each subplot pick its own y-limits so curves are not forced to share an axis range.
    shared_ylim = None

    for idx, (m, a, curves) in enumerate(entries):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        epochs, train_losses, val_losses = curves
        sns.lineplot(
            x=epochs,
            y=train_losses,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            linewidth=LINE_WIDTH,
            label="Train",
            ax=ax,
            color=PALETTE[0],
            alpha=0.9,
        )
        sns.lineplot(
            x=epochs,
            y=val_losses,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            linewidth=LINE_WIDTH,
            label="Val",
            ax=ax,
            color=PALETTE[1],
            alpha=0.9,
        )
        wrapped_title = textwrap.fill(f"{m.title()} with {augment_title(a)}{title_suffix}", width=28)
        ax.set_title(wrapped_title, pad=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if LOG_Y_SCALE:
            ax.set_yscale("log")
        ax.legend(frameon=True, fontsize="small")

    for ax in axes.flatten()[n:]:
        ax.axis("off")

    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.65, wspace=0.35)
    fig.savefig(FIG_ROOT / f"{grid_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_ROOT / f"{grid_base}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid plot to {FIG_ROOT / (grid_base + '.png')} and .svg")


all_curves = {}
recomputed_curves = {}

# Prepare data loaders once
train_path = pick_existing(LOCAL_TRAIN, REMOTE_TRAIN)
val_path = pick_existing(LOCAL_VAL, REMOTE_VAL)
train_loader = DataLoader(QM7XDataset(train_path), batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(QM7XDataset(val_path), batch_size=BATCH_SIZE, shuffle=False)

for m in MODEL_TYPES:
    for a in AUGMENT_TYPES:
        ckpt_dir = checkpoint_dir(m, a)
        skip_epochs = SKIP_EPOCHS.get(m, 0)
        curves = load_losses(ckpt_dir, skip_epochs=skip_epochs)
        if curves is None:
            print(f"Skipping {m}/{a}: no checkpoints in {ckpt_dir}")
            continue
        all_curves[(m, a)] = curves
        recomputed = recompute_losses(ckpt_dir, m, train_loader, val_loader, skip_epochs=skip_epochs) if PLOT_RECOMPUTED else None
        if recomputed is not None:
            recomputed_curves[(m, a)] = recomputed
        title = f"{m.title()} with {augment_title(a)}"
        if PLOT_LOGGED:
            outfile_base = f"loss_curve_{m}_{a}_logged"
            plot_single(*curves, title=title, outfile_base=outfile_base)
            print(f"Saved {outfile_base}.png/.svg in {FIG_ROOT}")
        if recomputed is not None:
            outfile_base_re = f"loss_curve_{m}_{a}_recomputed"
            plot_single(*recomputed, title=title, outfile_base=outfile_base_re)
            print(f"Saved {outfile_base_re}.png/.svg in {FIG_ROOT}")

if PLOT_LOGGED and all_curves:
    plot_grid(all_curves, grid_base="loss_curves_grid")

if PLOT_RECOMPUTED and recomputed_curves:
    plot_grid(recomputed_curves, grid_base="loss_curves_grid_recomputed", title_suffix=" (recomputed)")

if not all_curves and not recomputed_curves:
    print("No curves plotted; no checkpoints found for any model/augment combination.")

# ---------------------------------------------------------
# Predicted vs. true dipole components scatter (2x2 grid)
# ---------------------------------------------------------
pred_entries = []
for model_type, augment_type, label in PRED_SCATTER_ORDER:
    ckpt_dir = checkpoint_dir(model_type, augment_type)
    ckpt_path = latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        print(f"Skipping {label}: no checkpoint found in {ckpt_dir}")
        pred_entries.append(None)
        continue
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        print(f"Skipping {label}: checkpoint {ckpt_path} missing model_state_dict")
        pred_entries.append(None)
        continue
    model = build_model(model_type).to(DEVICE)
    model.load_state_dict(state_dict)
    true_arr, pred_arr, rmse = collect_pred_true(model, val_loader)
    if true_arr is None:
        print(f"Skipping {label}: no predictions collected")
        pred_entries.append(None)
        continue
    pred_entries.append((label, true_arr, pred_arr, rmse))

if any(entry is not None for entry in pred_entries):
    # Ensure we always have 4 slots for the 2x2 grid
    while len(pred_entries) < 4:
        pred_entries.append(None)
    plot_pred_true_grid(pred_entries)
else:
    print("No prediction-vs-true plots generated (no checkpoints with predictions).")
