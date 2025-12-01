import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import TRAIN_DATA, VAL_DATA, TEST_DATA

def scan(path: str):
    with h5py.File(path, "r") as f:
        dip = f["dipole"][:]
        lengths = f["length"][:]
    norms = np.linalg.norm(dip, axis=1)   # per-conformer magnitudes
    print(f"{path}: min dipole = {norms.min():.6f}, max dipole = {norms.max():.6f}, mean = {norms.mean():.6f}")
    print(f"{path}: min length = {lengths.min():.6f}, max length = {lengths.max():.6f}, mean = {lengths.mean():.6f}")
    for idx, axis in enumerate("xyz"):
        comp = dip[:, idx]
        print(f"  {axis}-component -> min: {comp.min():.6f}, max: {comp.max():.6f}, mean: {comp.mean():.6f}")
    counts, edges = np.histogram(norms, bins=20)
    print("Dipole magnitude histogram (count | bin_start -> bin_end):")
    for count, start, end in zip(counts, edges[:-1], edges[1:]):
        print(f"  {count:6d} | {start:.3f} -> {end:.3f}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.histplot(norms, bins=20, color="steelblue", edgecolor="black")
    plt.xlabel("Dipole Magnitude (Debye)")
    plt.ylabel("Count")
    plt.title(f"Dipole Histogram: {path}")
    plt.tight_layout()
    out_path = f"{path}_dipole_hist.png".replace("/", "_")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved histogram to {out_path}")
    
    mean_vec = dip.mean(axis=0)
    mse_mean = np.mean(np.sum((dip - mean_vec)**2, axis=1))
    print(f"Predict-mean baseline -> mean vector {mean_vec}, MSE {mse_mean:.6f}")



scan(TRAIN_DATA)
