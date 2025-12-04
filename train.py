import torch
import matplotlib.pyplot as plt
import seaborn as sns

from utils.rotation_utils import superfibonacci_rotations
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops

from qm7x_dataset import QM7XDataset
from models.equivariant import EquivariantModel
from models.gnn_nonequivariant import NonEquivariantModel
from models.vanilla import Vanilla

from tqdm import tqdm

from config.settings import (
    DEVICE,
    SEED,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    HIDDEN_DIM,
    N_LAYERS,
    TRAIN_DATA,
    VAL_DATA,
    NUM_EPOCHS,
    SHUFFLE,
    CHECKPOINT_PATH,
    FIG_PATH,
    CUTOFF,
    EQUIVARIANT,
    RESUME_PATH,
    N_ROTATIONS_EVALUATION,
)

torch.manual_seed(SEED)

train_dataset = QM7XDataset(TRAIN_DATA)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

val_dataset = QM7XDataset(VAL_DATA)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

if EQUIVARIANT:
    model = EquivariantModel(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)
else:
    model = Vanilla(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)

# model = NonEquivariantModel(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
mse = torch.nn.MSELoss()

start_epoch = 1

# ---------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------

def complete_graph(num_nodes, device='cpu'):
    idx = torch.arange(num_nodes, device=device)
    row = idx.repeat_interleave(num_nodes)
    col = idx.repeat(num_nodes)
    return torch.stack([row, col], dim=0)


def build_block_complete_graph(batch_indices):
    """Construct molecule-wise complete graphs to avoid cross-graph edges."""
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


# ---------------------------------------------------------------------
# Core training utilities
# ---------------------------------------------------------------------

def compute_batch_loss(batch):
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
    return loss


def evaluate(loader):
    model.eval()
    losses = []

    with torch.no_grad():
        print("Evaluating on validation set...")
        for batch in tqdm(loader):
            loss = compute_batch_loss(batch)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def rotate_evaluate(loader, n_rotations=10):
    model.eval()
    losses = []

    with torch.no_grad():
        print("Evaluating on validation set with rotations...")
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

            # Sample quasi-uniform rotations
            rotations = superfibonacci_rotations(
                n_rotations, device=pos.device, dtype=pos.dtype
            )

            batch_loss = 0.0

            for R in rotations:
                rotated_pos = pos @ R.T
                rotated_dip = dip @ R.T
                pred, _ = model(z=z, pos=rotated_pos, edge_index=edge_index, batch=b)
                loss = mse(pred, rotated_dip)
                batch_loss += loss.item()

            batch_loss /= n_rotations
            losses.append(batch_loss)

    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------
# Resume checkpoint (if provided)
# ---------------------------------------------------------------------

if RESUME_PATH:
    checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

    print(f"Resuming from {RESUME_PATH} (epoch {checkpoint['epoch']})")


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

train_epoch_losses = []
val_epoch_losses = []
rotate_val_epoch_losses = []

for epoch in range(start_epoch, NUM_EPOCHS + 1):
    loss_list = []

    for batch in tqdm(train_loader):
        model.train()
        loss = compute_batch_loss(batch)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = sum(loss_list) / len(loss_list)
    train_epoch_losses.append(avg_loss)

    val_loss = evaluate(val_loader)
    rotate_val_loss = rotate_evaluate(val_loader, n_rotations=N_ROTATIONS_EVALUATION)

    val_epoch_losses.append(val_loss)
    rotate_val_epoch_losses.append(rotate_val_loss)

    scheduler.step()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
        'val_loss': val_loss,
        'rotate_val_loss': rotate_val_loss,
    }

    torch.save(checkpoint, f'{CHECKPOINT_PATH}/checkpoint_epoch_{epoch}.pt')

    print(
        f"Epoch {epoch} | "
        f"Train Loss = {avg_loss:.6f} | "
        f"Val Loss = {val_loss:.6f} | "
        f"Rotated Val Loss = {rotate_val_loss:.6f}"
    )


# ---------------------------------------------------------------------
# Plot curves
# ---------------------------------------------------------------------

if train_epoch_losses:
    sns.set_theme(style="darkgrid")
    epochs = list(range(1, len(train_epoch_losses) + 1))

    plt.figure(figsize=(8, 4))
    sns.lineplot(x=epochs, y=train_epoch_losses, marker='o', label='Train Loss')

    if val_epoch_losses and len(val_epoch_losses) == len(train_epoch_losses):
        sns.lineplot(x=epochs, y=val_epoch_losses, marker='o', label='Val Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/training_loss_curve_.png")
    plt.close()

if rotate_val_epoch_losses:
    sns.set_theme(style="darkgrid")
    epochs = list(range(1, len(rotate_val_epoch_losses) + 1))

    plt.figure(figsize=(8, 4))
    sns.lineplot(
        x=epochs,
        y=rotate_val_epoch_losses,
        marker='o',
        label='Rotated Val Loss'
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Rotated Validation Loss Curve")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/rotated_validation_loss_curve_.png")
    plt.close()