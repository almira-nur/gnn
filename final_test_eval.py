from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

from models.chocolate import Chocolate
from models.strawberry import Strawberry
from qm7x_dataset import QM7XDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
HIDDEN_DIM = 64
N_LAYERS = 3
LR = 0.001
DATASET_TAG = "mini_200_conf_qm7x_processed_train"
TEST_DATA = "data/mini_200_conf_qm7x_processed_test.h5"
CHECKPOINT_ROOT = Path("checkpoints")
FIG_ROOT = Path("figures")
FIG_ROOT.mkdir(exist_ok=True)

MODEL_TYPES = ["chocolate", "strawberry"]
AUGMENT_TYPES = ["none", "superfib_end", "superfib_intermediate"]


def complete_graph(num_nodes: int, device: torch.device):
    idx = torch.arange(num_nodes, device=device)
    row = idx.repeat_interleave(num_nodes)
    col = idx.repeat(num_nodes)
    return torch.stack([row, col], dim=0)


def build_block_complete_graph(batch_indices: torch.Tensor) -> torch.Tensor:
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


def checkpoint_dir(model_type: str, augment_type: str) -> Path:
    name = f"{DATASET_TAG}_{model_type}_{augment_type}_hd{HIDDEN_DIM}_nl{N_LAYERS}_bs{BATCH_SIZE}_lr{LR}"
    return CHECKPOINT_ROOT / name / "checkpoints"


def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    ckpt_paths = sorted(
        ckpt_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not ckpt_paths:
        return None
    return ckpt_paths[-1]


def evaluate_loader(loader, model):
    mse_fn = nn.MSELoss()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
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
            loss = mse_fn(pred, dip)
            losses.append(loss.item())
    model.train()
    mean_mse = sum(losses) / len(losses) if losses else float("nan")
    rmse = mean_mse ** 0.5
    return mean_mse, rmse


def main():
    dataset = QM7XDataset(TEST_DATA)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Evaluating on test set: {TEST_DATA}")
    summary: Dict[tuple, tuple] = {}

    for m in MODEL_TYPES:
        for a in AUGMENT_TYPES:
            ckpt_dir = checkpoint_dir(m, a)
            ckpt_path = latest_checkpoint(ckpt_dir)
            if ckpt_path is None:
                print(f"Skipping {m}/{a}: no checkpoints found in {ckpt_dir}")
                continue
            model = build_model(m).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            if "model_state_dict" not in ckpt:
                print(f"Skipping {m}/{a}: checkpoint missing model_state_dict ({ckpt_path})")
                continue
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"\n{m}/{a}: loaded {ckpt_path.name} (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')})")

            mse, rmse = evaluate_loader(loader, model)
            summary[(m, a)] = (mse, rmse)
            print(f"→ Test MSE: {mse:.6f} | RMSE: {rmse:.6f}")

    print("\n=== Summary (test set) ===")
    for (m, a), (mse, rmse) in summary.items():
        print(f"{m:10s} | {a:18s} | MSE={mse:.6f} | RMSE={rmse:.6f}")

    if summary:
        sns.set_theme(style="whitegrid", context="talk")
        labels = []
        rmses = []
        for (m, a), (_, rmse) in summary.items():
            labels.append(f"{m}-{a}")
            rmses.append(rmse)

        plt.figure(figsize=(9, 4))
        ax = sns.barplot(x=labels, y=rmses, palette="Set2")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Model / Augmentation")
        ax.set_title("Test RMSE by model")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        png_path = FIG_ROOT / "test_rmse_bar.png"
        svg_path = FIG_ROOT / "test_rmse_bar.svg"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.savefig(svg_path, bbox_inches="tight")
        plt.close()
        print(f"Saved RMSE bar plot to {png_path} and {svg_path}")


if __name__ == "__main__":
    main()
