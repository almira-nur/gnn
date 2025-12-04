# train_egnn_qm7_dipole.py
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from egnn_layers import EGNNDipoleModel
from qm7_dipole_dataset import QM7DipoleDataset, collate_qm7



def random_rotation_matrix(device):
    A = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    det = torch.sign(torch.linalg.det(Q))
    Q[:, 2] *= det  # enforce det=+1
    return Q  # [3,3]


def apply_rotation(pos, batch, y=None):
    """
    Rotate all molecules in a batch by a single random rotation.
    pos: [N,3], batch: [N]
    y: [G,3] or None (dipoles)
    """
    device = pos.device
    R = random_rotation_matrix(device)

    pos_rot = (R @ pos.T).T
    if y is None:
        return pos_rot, None
    y_rot = (R @ y.T).T
    return pos_rot, y_rot


# --------------------
# training script
# --------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = QM7DipoleDataset(
        db_path=args.db_path,
        dipole_key=args.dipole_key,
        max_mols=args.max_mols,
    )

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_qm7(b, cutoff=args.cutoff),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_qm7(b, cutoff=args.cutoff),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_qm7(b, cutoff=args.cutoff),
    )

    model = EGNNDipoleModel(
        num_atom_types=args.num_atom_types,
        in_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            z = batch["z"].to(device)
            pos = batch["pos"].to(device)
            y = batch["y"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_vec = batch["batch"].to(device)

            # (optional) random rotation augmentation with rotated labels
            if args.rotate_train:
                pos, y = apply_rotation(pos, batch_vec, y)

            pred = model(z, pos, edge_index, batch_vec)  # [G,3]
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)

        train_loss /= len(train_set)

        # ---- validation + rotation consistency ----
        model.eval()
        val_loss = 0.0
        val_rot_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                z = batch["z"].to(device)
                pos = batch["pos"].to(device)
                y = batch["y"].to(device)
                edge_index = batch["edge_index"].to(device)
                batch_vec = batch["batch"].to(device)

                # original
                pred = model(z, pos, edge_index, batch_vec)
                loss = criterion(pred, y)
                val_loss += loss.item() * y.size(0)

                # rotated inputs + rotated labels (SE(3)-equivariance check)
                pos_rot, y_rot = apply_rotation(pos, batch_vec, y)
                pred_rot = model(z, pos_rot, edge_index, batch_vec)
                loss_rot = criterion(pred_rot, y_rot)
                val_rot_loss += loss_rot.item() * y.size(0)

        val_loss /= len(val_set)
        val_rot_loss /= len(val_set)

        print(
            f"Epoch {epoch:03d} | "
            f"train {train_loss:.4e} | "
            f"val {val_loss:.4e} | "
            f"val_rot {val_rot_loss:.4e}"
        )

        # simple checkpointing
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.out_model)
            print(f"  -> new best model saved to {args.out_model}")

    print("Training done. Best val loss:", best_val)

    # (optional) quick test evaluation
    model.load_state_dict(torch.load(args.out_model, map_location=device))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            z = batch["z"].to(device)
            pos = batch["pos"].to(device)
            y = batch["y"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_vec = batch["batch"].to(device)

            pred = model(z, pos, edge_index, batch_vec)
            loss = criterion(pred, y)
            test_loss += loss.item() * y.size(0)
    test_loss /= len(test_set)
    print("Test MSE on dipoles:", test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, required=True,
                        help="Path to QM7(-X) ASE/SchnetPack DB (e.g. QM7X.db)")
    parser.add_argument("--dipole_key", type=str, default="vDIP",
                        help="Key name for dipole vector in the DB")
    parser.add_argument("--max_mols", type=int, default=50000,
                        help="Optional cap on number of molecules (for Colab).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--num_atom_types", type=int, default=100)
    parser.add_argument("--rotate_train", action="store_true",
                        help="Apply random rotations during training.")
    parser.add_argument("--out_model", type=str, default="egnn_qm7_dipole.pt")

    args = parser.parse_args()
    main(args)
