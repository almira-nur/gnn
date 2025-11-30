import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import complete_graph, remove_self_loops
from qm7x_dataset import QM7XDataset
from models.equivariant import EquivariantModel
from tqdm import tqdm
from config.settings import DEVICE, SEED, BATCH_SIZE, LR, HIDDEN_DIM, N_LAYERS, DATA_PATH, NUM_EPOCHS, SHUFFLE



torch.manual_seed(SEED)

dataset = QM7XDataset(DATA_PATH)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

model = EquivariantModel(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
mse = torch.nn.MSELoss()

for epoch in range(1, NUM_EPOCHS + 1):
    for batch in train_loader:
        batch = batch.to(DEVICE)
        z = batch.z
        pos = batch.pos
        dip = batch.dip
        b = batch.batch
        N = pos.size(0)
        edge_index = complete_graph(N, device=DEVICE)
        mask = (b[edge_index[0]] == b[edge_index[1]])

        edge_index, _ = remove_self_loops(edge_index[:, mask])

        pred = model(z=z, pos=pos, edge_index=edge_index, batch=b)

        loss = mse(pred, dip)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} | Loss = {loss.item():.6f}")


