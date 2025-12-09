from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import remove_self_loops
from torch.utils.data import Subset


from models.chocolate import Chocolate
from models.strawberry import Strawberry
from qm7x_dataset import QM7XDataset

sns.set_theme(style="whitegrid", context="talk")

MODEL_TYPES = ["chocolate", "strawberry"]
AUGMENT_TYPES = ["none", "superfib_end", "superfib_intermediate"]
HIDDEN_DIM = 64
N_LAYERS = 3
BATCH_SIZE = 64
LR = 0.001
DATASET_TAG = "mini_200_conf_qm7x_processed_train"  # to match checkpoint naming
CHECKPOINT_ROOT = Path("checkpoints")
FIG_ROOT = Path("figures")
FIG_ROOT.mkdir(exist_ok=True)
DEVICE = torch.device('cpu')

SMALL_DATASETS = [
    "data/mini_200_conf_qm7x_processed_val.h5",
    "data/mini_200_conf_qm7x_processed_train.h5",
]
MAX_CONFS = 200
PROJECTION_METHOD = "pca"
PROJECTION_DIM = 2
COLOR_BY = "target_norm"


def pick_existing(*paths: str) -> str:
    for p in paths:
        if p and Path(p).exists():
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")


def complete_graph(num_nodes: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(num_nodes, device=device)
    row = idx.repeat_interleave(num_nodes)
    col = idx.repeat(num_nodes)
    return torch.stack([row, col], dim=0)


def build_block_complete_graph(batch_indices: torch.Tensor) -> torch.Tensor:
    device = batch_indices.device
    edge_chunks: List[torch.Tensor] = []
    for mol_id in batch_indices.unique(sorted=True):
        node_idx = torch.nonzero(batch_indices == mol_id, as_tuple=False).view(-1)
        if node_idx.numel() == 0:
            continue
        local_edges = complete_graph(node_idx.numel(), device=device)
        edge_chunks.append(node_idx[local_edges])
    if edge_chunks:
        return torch.cat(edge_chunks, dim=1)
    return torch.empty((2, 0), dtype=torch.long, device=device)


def build_model(model_type: str, hidden_dim: int, n_layers: int):
    model_type = model_type.lower()
    if model_type == "chocolate":
        return Chocolate(hidden_dim=hidden_dim, n_layers=n_layers)
    if model_type == "strawberry":
        return Strawberry(hidden_dim=hidden_dim, n_layers=n_layers)
    raise ValueError(f"Unknown model type: {model_type}")


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} missing model_state_dict; found keys {list(ckpt.keys())}")
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def pool_graph_embedding(v: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    # v shape: (N, 3, F) -> flatten, pool, reshape back to (B, 3*F)
    v_flat = v.view(v.size(0), -1)
    pooled = global_add_pool(v_flat, batch)
    return pooled  # (B, 3*F)


def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_index: int = -1,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    embeddings: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    preds: List[np.ndarray] = []

    is_chocolate = isinstance(model, Chocolate)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(loader), desc="Extracting embeddings", leave=False):
            batch = batch.to(device)
            z, pos, dip, b = batch.z, batch.pos, batch.dip, batch.batch
            edge_index = build_block_complete_graph(b)
            edge_index, _ = remove_self_loops(edge_index)

            captured: Dict[str, torch.Tensor] = {}
            hook = None
            if is_chocolate:
                def _hook(_module, _inputs, output):
                    # output is (x, v)
                    captured["v"] = output[1]
                hook = model.layers[layer_index].register_forward_hook(_hook)

            pred, intermediates = model(z=z, pos=pos, edge_index=edge_index, batch=b)

            if hook is not None:
                hook.remove()

            if isinstance(intermediates, list) and intermediates:
                v = intermediates[layer_index]
            elif is_chocolate and "v" in captured:
                v = captured["v"]
            else:
                raise RuntimeError("Could not access latent vectors from the model.")

            graph_emb = pool_graph_embedding(v, b)

            embeddings.append(graph_emb.detach().cpu().numpy())
            targets.append(dip.view(-1, 3).detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    model.train()
    emb_arr = np.concatenate(embeddings, axis=0)
    target_arr = np.concatenate(targets, axis=0)
    pred_arr = np.concatenate(preds, axis=0)
    return emb_arr, target_arr, pred_arr


def project_embeddings(
    emb: np.ndarray,
    method: str = "umap",
    dim: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=dim, random_state=random_state)
        return reducer.fit_transform(emb)
    if method == "tsne":
        reducer = TSNE(n_components=dim, random_state=random_state, init="pca", learning_rate="auto")
        return reducer.fit_transform(emb)
    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise ImportError("umap-learn is required for method='umap'. Install with `pip install umap-learn`.") from exc
        reducer = umap.UMAP(n_components=dim, random_state=random_state)
        return reducer.fit_transform(emb)
    raise ValueError(f"Unknown projection method: {method}")


def make_scatter(
    proj: np.ndarray,
    color: Optional[np.ndarray],
    title: str,
    outfile: Path,
    color_label: str = "Color value",
):
    dim = proj.shape[1]
    cmap = plt.cm.viridis
    fig = plt.figure(figsize=(7, 6))
    if dim == 3:
        ax: Axes = fig.add_subplot(111, projection="3d")  # type: ignore
        sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=color, cmap=cmap, s=12, alpha=0.9)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    else:
        ax: Axes = fig.add_subplot(111)
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=color, cmap=cmap, s=12, alpha=0.9)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    ax.set_title(title)
    if color is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label(color_label)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved projection to {outfile}")


def choose_color(values: np.ndarray, mode: str) -> Optional[np.ndarray]:
    if mode == "none":
        return None
    if mode == "target_norm":
        return np.linalg.norm(values, axis=1)
    if mode == "pred_norm":
        return np.linalg.norm(values, axis=1)
    if mode == "target_x":
        return values[:, 0]
    if mode == "target_y":
        return values[:, 1]
    if mode == "target_z":
        return values[:, 2]
    raise ValueError(f"Unknown color mode: {mode}")


def augment_title(augment_type: str) -> str:
    match augment_type:
        case "none":
            return "No Augmentation"
        case "superfib_end":
            return "Augmentation"
        case "superfib_intermediate":
            return "Augmentation + Layerwise Loss"
        case _:
            return augment_type


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


def main():
    dataset_path = pick_existing(*SMALL_DATASETS)
    base_dataset = QM7XDataset(dataset_path)
    dataset = Subset(base_dataset, range(min(len(base_dataset), MAX_CONFS)))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    for model_type in MODEL_TYPES:
        for augment_type in AUGMENT_TYPES:
            ckpt_dir = checkpoint_dir(model_type, augment_type)
            ckpt_path = latest_checkpoint(ckpt_dir)
            if ckpt_path is None:
                print(f"Skipping {model_type}/{augment_type}: no checkpoints in {ckpt_dir}")
                continue

            model = build_model(model_type, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)
            ckpt = load_checkpoint(model, ckpt_path, DEVICE)
            print(f"Loaded {ckpt_path} for {model_type}/{augment_type} (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')})")

            emb, target, pred = extract_embeddings(
                model,
                loader,
                device=DEVICE,
                layer_index=-1,
                max_batches=None,
            )

            color_source = target if COLOR_BY.startswith("target") else pred
            color_vals = choose_color(color_source, COLOR_BY)
            proj = project_embeddings(emb, method=PROJECTION_METHOD, dim=PROJECTION_DIM)

            title = f"{model_type.title()} with {augment_title(augment_type)}"
            out_name = f"latent_{PROJECTION_METHOD}_{model_type}_{augment_type}_{ckpt_path.stem}.png"
            outfile = FIG_ROOT / out_name
            make_scatter(proj, color_vals, title=title, outfile=outfile, color_label=COLOR_BY)


if __name__ == "__main__":
    main()
