import math
import textwrap
import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from torch.utils.data import Subset
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
FIG_ROOT = Path("figures")
FIG_ROOT.mkdir(exist_ok=True)

N_ROTATIONS = 64   # how many SO(3) samples to test
BATCH_SIZE = 1     # process one molecule per batch for clean equivariance eval
MAX_MOLECULES = 256  # set to None to use all; lower to speed up evaluation

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def checkpoint_dir(model_type: str, augment_type: str) -> Path:
    name = f"{DATASET_TAG}_{model_type}_{augment_type}_hd{HIDDEN_DIM}_nl{N_LAYERS}_bs64_lr{LR}"
    return CHECKPOINT_ROOT / name / "checkpoints"


def augment_title(augment_type: str) -> str:
    match augment_type:
        case "none":
            return "no augmentation"
        case "superfib_end":
            return "augmentation"
        case "superfib_intermediate":
            return "augmentation with layerwise loss"
        case _:
            return augment_type.replace("_", " ")


def display_model_label(model_type: str, augment_type: str) -> str:
    if augment_type == "none":
        if model_type == "chocolate":
            return "Chocolate"
        if model_type == "strawberry":
            return "Vanilla"
    base = model_type.title()
    aug_text = augment_title(augment_type)
    return f"{base} with {aug_text}"


def slugify_label(model_type: str, augment_type: str) -> str:
    label = display_model_label(model_type, augment_type)
    return label.lower().replace(" ", "_")


def flavor_color(model_type: str, augment_type: str) -> str:
    # Choose colors by flavor metaphor
    if model_type == "chocolate":
        return "#8B4513"  # saddle brown
    if model_type == "strawberry":
        if augment_type == "none":
            return "#F5E6C8"  # vanilla cream
        return "#F06292"      # strawberry pink
    return "#4C6FBF"          # fallback blue


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
    Returns:
        layer_means: mean equivariance error per layer
        errors_per_layer: list of lists of errors over rotations
        norms_per_layer: mean vector norm per layer (base orientation), one scalar
    """
    graph = graph.to(DEVICE)
    z, pos, batch = graph.z, graph.pos, graph.batch
    edge_index = build_edges(batch)
    edge_index, _ = remove_self_loops(edge_index)

    # ---- baseline: vectors at identity ----
    v_layers_base = get_layerwise_vectors(model, z, pos, edge_index, batch)

    errors_per_layer = [[] for _ in range(len(v_layers_base))]
    norms_per_layer = []

    # collect baseline norms (mean vector magnitude per layer)
    for v0 in v_layers_base:
        vectors = v0.permute(0, 2, 1).reshape(-1, 3)  # (N*F, 3)
        norms_per_layer.append(torch.norm(vectors, dim=1).mean().item())

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
    layer_means = [float(np.mean(errs)) for errs in errors_per_layer]
    return layer_means, errors_per_layer, norms_per_layer


# ---------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------

def plot_violin(errors_per_layer, model_type: str, augment_type: str):
    layers, errs = [], []
    for layer_idx, vals in enumerate(errors_per_layer):
        layers.extend([layer_idx] * len(vals))
        errs.extend(vals)

    if not errs:
        print(f"No errors to plot for {model_type}/{augment_type}")
        return

    plt.figure(figsize=(7, 4.5))
    sns.violinplot(
        x=layers,
        y=errs,
        inner="box",
        cut=0,
        scale="width",
        color=flavor_color(model_type, augment_type),
        saturation=1.0,
    )
    plt.xlabel("Layer")
    plt.ylabel("Relative equivariance error")
    wrapped = textwrap.fill(
        f"Equivariance over rotations: {display_model_label(model_type, augment_type)}",
        width=38,
    )
    plt.title(wrapped)
    ymin, ymax = min(errs), max(errs)
    pad = 0.05 * (ymax - ymin + 1e-12)
    plt.ylim(ymin - pad, ymax + pad)
    plt.tight_layout()

    base = FIG_ROOT / f"equivariance_violin_{slugify_label(model_type, augment_type)}"
    plt.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close()
    print(f"Saved violin plot to {base.with_suffix('.png')} and .svg")


def plot_norm_violin(norms_per_layer, model_type: str, augment_type: str):
    layers = []
    norms = []
    for layer_idx, vals in enumerate(norms_per_layer):
        layers.extend([layer_idx] * len(vals))
        norms.extend(vals)

    if not norms:
        print(f"No norms to plot for {model_type}/{augment_type}")
        return

    plt.figure(figsize=(7, 4.5))
    sns.violinplot(
        x=layers,
        y=norms,
        inner="box",
        cut=0,
        scale="width",
        color=flavor_color(model_type, augment_type),
        saturation=1.0,
    )
    plt.xlabel("Layer")
    plt.ylabel("Vector norm (mean |v|)")
    wrapped = textwrap.fill(
        f"Layer vector norms: {display_model_label(model_type, augment_type)}",
        width=38,
    )
    plt.title(wrapped)
    ymin, ymax = min(norms), max(norms)
    pad = 0.05 * (ymax - ymin + 1e-12)
    plt.ylim(ymin - pad, ymax + pad)
    plt.tight_layout()

    base = FIG_ROOT / f"equivariance_norms_{slugify_label(model_type, augment_type)}"
    plt.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close()
    print(f"Saved norm plot to {base.with_suffix('.png')} and .svg")


def plot_violin_grid(all_distributions: dict[tuple[str, str], list[list[float]]]):
    entries = list(all_distributions.items())
    if not entries:
        print("No violin data to grid-plot.")
        return

    n = len(entries)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.0 * ncols, 4.2 * nrows),
        squeeze=False,
    )

    # Collect limits to synchronize y-axis across all subplots
    all_errs = []
    for idx, ((model_type, augment_type), layer_vals) in enumerate(entries):
        layers, errs = [], []
        for layer_idx, vals in enumerate(layer_vals):
            layers.extend([layer_idx] * len(vals))
            errs.extend(vals)
        all_errs.extend(errs)
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sns.violinplot(
            x=layers,
            y=errs,
            inner="box",
            cut=0,
            scale="width",
            ax=ax,
            color=flavor_color(model_type, augment_type),
            saturation=1.0,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Relative equivariance error")
        wrapped = textwrap.fill(display_model_label(model_type, augment_type), width=26)
        ax.set_title(wrapped)

    # Hide unused axes if any
    for ax in axes.flatten()[n:]:
        ax.axis("off")

    if all_errs:
        ymin, ymax = min(all_errs), max(all_errs)
        pad = 0.05 * (ymax - ymin + 1e-12)
        shared_ylim = (ymin - pad, ymax + pad)
        for ax in axes.flatten()[:n]:
            ax.set_ylim(*shared_ylim)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.85, wspace=0.45, top=0.90)
    base = FIG_ROOT / "equivariance_violin_grid"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved violin grid to {base.with_suffix('.png')} and .svg")


def plot_norm_grid(all_norms: dict[tuple[str, str], list[list[float]]]):
    entries = list(all_norms.items())
    if not entries:
        print("No norm data to grid-plot.")
        return

    n = len(entries)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.0 * ncols, 4.2 * nrows),
        squeeze=False,
    )

    all_vals = []
    for idx, ((model_type, augment_type), layer_vals) in enumerate(entries):
        layers, norms = [], []
        for layer_idx, vals in enumerate(layer_vals):
            layers.extend([layer_idx] * len(vals))
            norms.extend(vals)
        all_vals.extend(norms)
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sns.violinplot(
            x=layers,
            y=norms,
            inner="box",
            cut=0,
            scale="width",
            ax=ax,
            color=flavor_color(model_type, augment_type),
            saturation=1.0,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Vector norm (mean |v|)")
        wrapped = textwrap.fill(display_model_label(model_type, augment_type), width=26)
        ax.set_title(wrapped)

    for ax in axes.flatten()[n:]:
        ax.axis("off")

    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        pad = 0.05 * (ymax - ymin + 1e-12)
        shared_ylim = (ymin - pad, ymax + pad)
        for ax in axes.flatten()[:n]:
            ax.set_ylim(*shared_ylim)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.85, wspace=0.45, top=0.90)
    base = FIG_ROOT / "equivariance_norms_grid"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved norm grid to {base.with_suffix('.png')} and .svg")


# ---------------------------------------------------------
# Main driver
# ---------------------------------------------------------

def main():
    # load dataset (optionally subset to speed up)
    dataset = QM7XDataset("data/mini_200_conf_qm7x_processed_val.h5")
    if MAX_MOLECULES is not None:
        max_len = min(len(dataset), MAX_MOLECULES)
        dataset = Subset(dataset, list(range(max_len)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    graphs = list(loader)

    rotations = superfibonacci_rotations(N_ROTATIONS, device=DEVICE, dtype=torch.float32)

    all_results = {}
    all_distributions = {}
    all_norms = {}

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

            aggregated = None
            aggregated_norms = None
            for graph in graphs:
                layer_means_single, layer_distributions_single, layer_norms_single = compute_equivariance_error(model, graph, rotations)
                if aggregated is None:
                    aggregated = [[] for _ in layer_distributions_single]
                if aggregated_norms is None:
                    aggregated_norms = [[] for _ in layer_norms_single]
                for ℓ, vals in enumerate(layer_distributions_single):
                    aggregated[ℓ].extend(vals)
                for ℓ, norm_val in enumerate(layer_norms_single):
                    aggregated_norms[ℓ].append(norm_val)

            if aggregated is None:
                print(f"No data processed for {model_type}/{augment_type}")
                continue

            layer_means = [float(np.mean(vals)) if vals else float("nan") for vals in aggregated]

            all_results[(model_type, augment_type)] = layer_means
            all_distributions[(model_type, augment_type)] = aggregated
            all_norms[(model_type, augment_type)] = aggregated_norms

            plot_violin(aggregated, model_type, augment_type)
            plot_norm_violin(aggregated_norms, model_type, augment_type)

            for i, err in enumerate(layer_means):
                print(f"  Layer {i}: equivariance error = {err:.3e}")

    print("\n=== Equivariance Summary ===")
    for (model_type, aug), errs in all_results.items():
        print(f"{model_type:10s} | {aug:20s} | " +
              " ".join([f"L{i}:{e:.2e}" for i, e in enumerate(errs)]))

    plot_violin_grid(all_distributions)
    plot_norm_grid(all_norms)


if __name__ == "__main__":
    main()
