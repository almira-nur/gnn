import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

from qm7x_dataset import QM7XDataset
from utils.rotation_utils import superfibonacci_rotations
from models.chocolate import Chocolate
from models.strawberry import Strawberry

# ========= SETTINGS ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

HIDDEN_DIM = 64
N_LAYERS = 3

DATASET_PATH = "data/mini_200_conf_qm7x_processed_val.h5"

CHECKPOINT_ROOT = Path("checkpoints")
MODEL_TYPES = ["chocolate", "strawberry"]
AUGMENT_TYPES = ["none", "superfib_end", "superfib_intermediate"]

N_ROTATIONS_EVAL = 32
EPS = 1e-8

# =============================================================================
# Utility: Build a block-complete graph (same function as your training script)
# =============================================================================

def complete_graph(num_nodes, device="cpu"):
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

# =============================================================================
# Model loading
# =============================================================================

def build_model(model_type: str):
    if model_type == "chocolate":
        return Chocolate(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    elif model_type == "strawberry":
        return Strawberry(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    else:
        raise ValueError(f"Unknown model type {model_type}")

def latest_checkpoint(model_type: str, augment_type: str):
    pattern = f"mini_200_conf_qm7x_processed_train_{model_type}_{augment_type}_hd{HIDDEN_DIM}_nl{N_LAYERS}_bs64_lr0.001"
    ckpt_dir = CHECKPOINT_ROOT / pattern / "checkpoints"
    if not ckpt_dir.exists():
        return None

    ckpts = sorted(
        ckpt_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    return ckpts[-1] if ckpts else None

# =============================================================================
# Dipole-level equivariance error
# =============================================================================

def dipole_equivariance_error(model, loader, n_rotations=N_ROTATIONS_EVAL):
    model.eval()
    errors = []

    with torch.no_grad():
        rotations = superfibonacci_rotations(n_rotations, device=DEVICE, dtype=torch.float32)

        print("Evaluating dipole equivariance error...")
        for batch in tqdm(loader):
            batch = batch.to(DEVICE)
            z, pos, b = batch.z, batch.pos, batch.batch

            # Build edges
            edge_index = build_block_complete_graph(b)
            edge_index, _ = remove_self_loops(edge_index)

            # Base prediction
            pred_base, _ = model(z=z, pos=pos, edge_index=edge_index, batch=b)  # (B,3)

            for R in rotations:
                rot_pos = pos @ R.T

                pred_rot, _ = model(z=z, pos=rot_pos, edge_index=edge_index, batch=b)

                pred_expected = pred_base @ R.T

                # Relative equivariance error
                diff = torch.norm(pred_rot - pred_expected, dim=1)              # (B,)
                base_norm = torch.norm(pred_base, dim=1) + EPS                  # (B,)
                rel_error = (diff / base_norm).mean().item()
                errors.append(rel_error)

    return sum(errors) / len(errors) if errors else float("nan")

# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading dataset:", DATASET_PATH)
    dataset = QM7XDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n=== EQUIVARIANCE RESULTS ===\n")
    results = {}

    for model_type in MODEL_TYPES:
        for aug in AUGMENT_TYPES:

            ckpt_path = latest_checkpoint(model_type, aug)
            if ckpt_path is None:
                print(f"Skipping {model_type}/{aug}: no checkpoint found.")
                continue

            print(f"\nLoading {model_type}/{aug} from {ckpt_path}")
            model = build_model(model_type).to(DEVICE)

            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])

            eq_err = dipole_equivariance_error(model, loader, n_rotations=N_ROTATIONS_EVAL)

            print(f"→ Dipole equivariance error: {eq_err:.3e}")
            results[(model_type, aug)] = eq_err

    print("\n=== Summary ===")
    for (m, a), err in results.items():
        print(f"{m:10s} | {a:18s} | equiv err = {err:.3e}")


if __name__ == "__main__":
    main()
