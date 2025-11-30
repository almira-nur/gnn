import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
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

# This code should work for both the equivariant and baseline models.
def complete_graph(num_nodes, device='cpu'):
    idx = torch.arange(num_nodes, device=device)
    row = idx.repeat_interleave(num_nodes)   
    col = idx.repeat(num_nodes)
    return torch.stack([row, col], dim=0)

for epoch in range(1, NUM_EPOCHS + 1):
    for batch in tqdm(train_loader):
        batch = batch.to(DEVICE)
        z = batch.z
        pos = batch.pos
        dip = batch.dip
        b = batch.batch
        N = pos.size(0)
        B = b.max().item() + 1

        dip = dip.view(B, 3)

        #Generate the graph and remove self-loops
        edge_index = complete_graph(N, device=DEVICE)
        mask = (b[edge_index[0]] == b[edge_index[1]])

        edge_index, _ = remove_self_loops(edge_index[:, mask])

        pred = model(z=z, pos=pos, edge_index=edge_index, batch=b)

        loss = mse(pred, dip)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}.pt')

    print(f"Epoch {epoch} | Loss = {loss.item():.6f}")


