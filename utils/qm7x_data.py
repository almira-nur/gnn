import h5py
import torch

from torch.utils.data import Dataset

class QM7X_Dataset(Dataset):
    def __init__(self, path):
        self.f = h5py.File(path, "r")

        self.Z = self.f["Z"]
        self.pos = self.f["pos"]
        self.start = self.f["start"]
        self.length = self.f["length"]

        self.e_pbe0 = self.f["e_pbe0"]
        self.dip = self.f["dipole"]
        self.quadrupole = self.f["quadrupole"]
        self.pol = self.f["polarizability"]
        

        self.N = len(self.start)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        s = int(self.start[idx])
        n = int(self.length[idx])
        e = s + n

        data = {
            "z":   torch.tensor(self.Z[s:e], dtype=torch.long),
            "pos": torch.tensor(self.pos[s:e], dtype=torch.float32),
            "e_pbe0": torch.tensor(self.e_pbe0[idx:idx+1], dtype=torch.float32),
            "dipole": torch.tensor(self.dip[idx], dtype=torch.float32),
            "quadrupole": torch.tensor(self.quad[idx], dtype=torch.float32),
            "polarizability": torch.tensor(self.pol[idx], dtype=torch.float32),
        }

        return data
