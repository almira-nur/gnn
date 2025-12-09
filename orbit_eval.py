import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from torch.utils.data import Subset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from qm7x_dataset import QM7XDataset
from models.chocolate import Chocolate
from models.strawberry import Strawberry
from utils.rotation_utils import superfibonacci_rotations


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TYPES = ["chocolate", "strawberry"]
AUGMENT_TYPES = ["none", "superfib_end", "superfib_intermediate"]
HIDDEN_DIM = 64
N_LAYERS = 3
LR = 0.001
DATASET_TAG = "mini_200_conf_qm7x_processed_train"
CHECKPOINT_ROOT = Path("checkpoints")

N_ROTATIONS = 64   # how many SO(3) samples to test
BATCH_SIZE = 1     # process one molecule per batch for clean equivariance eval
MAX_MOLECULES = 256  # set to None to use all; lower to speed up evaluation
MAX_MOLECULES = 64  # set None to evaluate all molecules; lower to speed up


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def checkpoint_dir(model_type: str, augment_type: str) -> Path:
    name = f"{DATASET_TAG}_{model_type}_{augment_type}_hd{HIDDEN_DIM}_nl{N_LAYERS}_bs64_lr{LR}"
    return CHECKPOINT_ROOT / name / "checkpoints"


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(
        ckpt_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    return ckpts[-1] if ckpts else None


def build_model(model_type: str):
    if model_type == "chocolate":
        return Chocolate(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    elif model_type == "strawberry":
        return Strawberry(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    else:
        raise ValueError(model_type)


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def complete_graph(n, device):
    idx = torch.arange(n, device=device)
    row = idx.repeat_interleave(n)
    col = idx.repeat(n)
    return torch.stack([row, col], dim=0)


def build_edges(batch):
    node_idx = torch.nonzero(batch == 0, as_tuple=False).view(-1)
    edges = complete_graph(node_idx.numel(), batch.device)
    return node_idx[edges]


# ---------------------------------------------------------
# Extract vector features at every layer
# ---------------------------------------------------------

def get_layerwise_vectors(model, z, pos, edge_index, batch):
    """
    Returns:
        v_list: list of tensors of shape (N, 3, F) for each layer
    """
    model.eval()

    v_list = []

    # attach hooks
    hooks = []
    for l in range(len(model.layers)):
        def make_hook(layer_index):
            def hook(mod, inp, out):
                # Out = (x, v)
                v_list.append(out[1].detach())
            return hook
        hooks.append(model.layers[l].register_forward_hook(make_hook(l)))

    with torch.no_grad():
        model(z=z, pos=pos, edge_index=edge_index, batch=batch)

    # remove hooks
    for h in hooks:
        h.remove()

    return v_list  # list of length N_LAYERS, each (N, 3, F)


# ---------------------------------------------------------
# Equivariance error computation
# ---------------------------------------------------------

def compute_equivariance_error(model, graph, rotations):
    """
    graph: one PyG data object
    rotations: tensor of shape (K, 3, 3)
    """
    graph = graph.to(DEVICE)
    z, pos, batch = graph.z, graph.pos, graph.batch
    edge_index = build_edges(batch)
    edge_index, _ = remove_self_loops(edge_index)

    # ---- baseline: vectors at identity ----
    v_layers_base = get_layerwise_vectors(model, z, pos, edge_index, batch)

    errors_per_layer = [[] for _ in range(len(v_layers_base))]

    with torch.no_grad():
        for R in rotations:
            R = R.to(DEVICE)
            pos_rot = pos @ R.T

            # get rotated features
            v_layers_rot = get_layerwise_vectors(model, z, pos_rot, edge_index, batch)

            # compare layer-by-layer
            for ℓ, (v0, vR) in enumerate(zip(v_layers_base, v_layers_rot)):
                # v0, vR: (N, 3, F)

                # rotate baseline vectors: R @ v0
                v0_rot = torch.einsum("ij, njf -> nif", R, v0)

                num = torch.norm(vR - v0_rot).item()
                den = torch.norm(v0).item() + 1e-12
                errors_per_layer[ℓ].append(num / den)

    # average over rotations
    return [float(np.mean(errs)) for errs in errors_per_layer]


# ---------------------------------------------------------
# Main driver
# ---------------------------------------------------------

def main():
    # load dataset
    dataset = QM7XDataset("data/mini_200_conf_qm7x_processed_val.h5")
    if MAX_MOLECULES is not None:
        max_len = min(len(dataset), MAX_MOLECULES)
        dataset = Subset(dataset, list(range(max_len)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # rotation samples
    rotations = superfibonacci_rotations(N_ROTATIONS, device=DEVICE, dtype=torch.float32)

    # store results
    all_results = {}

    for model_type in MODEL_TYPES:
        for augment_type in AUGMENT_TYPES:
            ckpt_dir = checkpoint_dir(model_type, augment_type)
            ckpt_path = latest_checkpoint(ckpt_dir)

            if ckpt_path is None:
                print(f"Skipping {model_type}/{augment_type}: no checkpoint.")
                continue

            model = build_model(model_type).to(DEVICE)
            load_checkpoint(model, ckpt_path)

            print(f"Testing equivariance: {model_type}/{augment_type}")
            for graph in loader:
                layer_errors = compute_equivariance_error(model, graph, rotations)

                all_results[(model_type, augment_type)] = layer_errors

                for i, err in enumerate(layer_errors):
                    print(f"  Layer {i}: equivariance error = {err:.3e}")

    # ---- summary table ----
    print("\n=== Equivariance Summary ===")
    for (model_type, aug), errs in all_results.items():
        print(f"{model_type:10s} | {aug:20s} | " +
              " ".join([f"L{i}:{e:.2e}" for i, e in enumerate(errs)]))


if __name__ == "__main__":
    main()
