import torch
import matplotlib.pyplot as plt
import seaborn as sns

from utils.rotation_utils import superfibonacci_rotations
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops

from qm7x_dataset import QM7XDataset
#from models.equivariant import EquivariantModel
#from models.gnn_nonequivariant import NonEquivariantModel
from models.vanilla import Vanilla 
from models.chocolate import Chocolate
from models.strawberry import Strawberry
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
    RESUME_PATH,
    N_ROTATIONS_EVALUATION,
    MODEL_TYPE,
    LAMBDA_CONSISTENCY,
    AUGMENT_TYPE,
    N_ROTATIONS_TRAIN,
)

torch.manual_seed(SEED)

train_dataset = QM7XDataset(TRAIN_DATA)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

val_dataset = QM7XDataset(VAL_DATA)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


match MODEL_TYPE:
    case 'vanilla':
        model = torch.compile(Vanilla(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE))
    case 'chocolate':
        model = torch.compile(Chocolate(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE))

    case 'strawberry':
        model = torch.compile(Strawberry(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE))


optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
mse = torch.nn.MSELoss()

start_epoch = 1


if RESUME_PATH:
    checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

    print(f"Resuming from {RESUME_PATH} (epoch {checkpoint['epoch']})")


match AUGMENT_TYPE:
    case 'none':
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

    case 'superfib_end':
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

            rot_count = max(1, N_ROTATIONS_TRAIN)
            rotations = superfibonacci_rotations(
                rot_count, device=pos.device, dtype=pos.dtype
            )

            total_loss = 0.0
            for R in rotations:
                rotated_pos = pos @ R.T
                rotated_dip = dip @ R.T
                pred, _ = model(z=z, pos=rotated_pos, edge_index=edge_index, batch=b)
                total_loss = total_loss + mse(pred, rotated_dip)

            total_loss = total_loss / rot_count
            return total_loss  

    case 'superfib_intermediate':
        def _rotate_vector(v, R):
            v = v.view(v.size(0), 3, -1)
            return torch.matmul(R, v.transpose(1, 2)).transpose(1, 2)

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

            base_pred, base_inter = model(z=z, pos=pos, edge_index=edge_index, batch=b)
            base_loss = mse(base_pred, dip)

            rot_count = max(1, N_ROTATIONS_TRAIN)
            rotations = superfibonacci_rotations(
                rot_count, device=pos.device, dtype=pos.dtype
            )

            total_loss = base_loss
            for R in rotations:
                rotated_pos = pos @ R.T
                rotated_dip = dip @ R.T
                rot_pred, rot_inter = model(z=z, pos=rotated_pos, edge_index=edge_index, batch=b)
                total_loss = total_loss + mse(rot_pred, rotated_dip)

                cons = 0.0
                count = 0
                for base_item, rot_item in zip(base_inter, rot_inter):
                    v_base = base_item
                    v_rot = rot_item
                    v_exp = _rotate_vector(v_base, R)
                    #v_rot = v_rot.view(v_rot.size(0), 3, -1)
                    cons = cons + torch.mean(torch.norm(v_rot - v_exp, dim=1))
                    count += 1
                if count > 0:
                    total_loss = total_loss + LAMBDA_CONSISTENCY * (cons / count)

            total_loss = total_loss / (rot_count + 1)
            return total_loss


def complete_graph(num_nodes, device='cpu'):
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

def evaluate(loader, name="validation"):
    model.eval()
    losses = []

    with torch.no_grad():
        print(f"Evaluating on {name} set...")
        for batch in tqdm(loader):
            loss = compute_batch_loss(batch)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

def rotate_evaluate(loader, n_rotations=N_ROTATIONS_EVALUATION):
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

def equivariance_error(loader, n_rotations=N_ROTATIONS_EVALUATION):
    model.eval()
    errors = []
    with torch.no_grad():
        print("Computing equivariance error on validation set...")
        for batch in tqdm(loader):
            batch = batch.to(DEVICE)
            z = batch.z
            pos = batch.pos
            b = batch.batch
            B = b.max().item() + 1

            edge_index = build_block_complete_graph(b)
            edge_index, _ = remove_self_loops(edge_index)

            # Sample quasi-uniform rotations on SO(3) to probe equivariance.
            rotations = superfibonacci_rotations(n_rotations, device=pos.device, dtype=pos.dtype)

            for R in rotations:
                rotated_pos = pos @ R.T

                pred, _ = model(z=z, pos=pos, edge_index=edge_index, batch=b)
                rotated_pred, _ = model(z=z, pos=rotated_pos, edge_index=edge_index, batch=b)

                rotated_pred_expected = pred @ R.T

                error = torch.mean(torch.norm(rotated_pred - rotated_pred_expected, dim=1)).item()
                errors.append(error)

    model.train()
    return sum(errors) / len(errors)






train_epoch_losses = []
train_eval_epoch_losses = []
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

#    train_eval_loss = evaluate(train_loader, name="train (eval)")
#    train_eval_epoch_losses.append(train_eval_loss)

    val_loss = evaluate(val_loader, name="validation")
#    rotate_val_loss = rotate_evaluate(val_loader, n_rotations=N_ROTATIONS_EVALUATION)

    val_epoch_losses.append(val_loss)
#    rotate_val_epoch_losses.append(rotate_val_loss)

    scheduler.step()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
#        'train_eval_loss': train_eval_loss,
        'val_loss': val_loss,
#        'rotate_val_loss': rotate_val_loss,
    }

    torch.save(checkpoint, f'{CHECKPOINT_PATH}/checkpoint_epoch_{epoch}.pt')

    print(f"Epoch {epoch} | Train Loss = {avg_loss:.6f} | Val Loss = {val_loss:.6f}")
    #print(f"Epoch {epoch} | Train Loss = {avg_loss:.6f} | Train Eval Loss = {train_eval_loss:.6f} | Val Loss = {val_loss:.6f} | Rotated Val Loss = {rotate_val_loss:.6f} | Equivariance Error = {equivariance_error(val_loader, n_rotations=N_ROTATIONS_EVALUATION):.6f}")

if train_epoch_losses:
    sns.set_theme(style="darkgrid")
    epochs = list(range(1, len(train_epoch_losses) + 1))

    plt.figure(figsize=(8, 4))
    sns.lineplot(x=epochs, y=train_epoch_losses, marker='o', label='Train Loss')
    if train_eval_epoch_losses and len(train_eval_epoch_losses) == len(train_epoch_losses):
        sns.lineplot(x=epochs, y=train_eval_epoch_losses, marker='o', label='Train Eval Loss')

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
