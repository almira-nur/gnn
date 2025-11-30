import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from qm7x_dataset import QM7XDataset
from models.equivariant import EquivariantModel
from tqdm import tqdm
from config.settings import DEVICE, SEED, BATCH_SIZE, LR, HIDDEN_DIM, N_LAYERS, DATA_PATH, NUM_EPOCHS, SHUFFLE, CHECKPOINT_PATH

torch.manual_seed(SEED)
dataset = QM7XDataset(DATA_PATH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
model = EquivariantModel(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
mse = torch.nn.MSELoss()