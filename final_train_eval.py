from pathlib import Path
from typing import Dict, Optional

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
TRAIN_DATA = "data/mini_200_conf_qm7x_processed_train.h5"

CHECKPOINT_ROOT = Path("checkpoints")

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
    mse = nn.MSELoss()
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
            loss = mse(pred, dip)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def main():
    dataset = QM7XDataset(TRAIN_DATA)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Evaluating on training set: {TRAIN_DATA}")
    summary: Dict[tuple, float] = {}

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

            loss = evaluate_loader(loader, model)
            summary[(m, a)] = loss
            print(f"→ Train MSE: {loss:.6f}")

    print("\n=== Summary (train MSE) ===")
    for (m, a), l in summary.items():
        print(f"{m:10s} | {a:18s} | {l:.6f}")


if __name__ == "__main__":
    main()
