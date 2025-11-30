import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class QM7XDataset(Dataset):

    def __init__(self, h5_path, target="dipole", device="cpu"):
        super().__init__()
        self.h5_path = h5_path
        self.target = target
        self.device = device

        # Read metadata only (fast)
        with h5py.File(h5_path, "r") as f:
            self.starts = torch.tensor(f["start"][:], dtype=torch.long)
            self.lengths = torch.tensor(f["length"][:], dtype=torch.long)
            self.num_confs = len(self.starts)

    def __len__(self):
        return self.num_confs

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            s = int(self.starts[idx])
            n = int(self.lengths[idx])

            # atomic numbers & Cartesian coordinates
            z = torch.tensor(f["z"][s:s+n], dtype=torch.long)
            pos = torch.tensor(f["pos"][s:s+n], dtype=torch.float32)

            # dipole (3,)
            dip = torch.tensor(f["dipole"][idx], dtype=torch.float32)

        data = Data(z=z, pos=pos, dip=dip)

        return data
