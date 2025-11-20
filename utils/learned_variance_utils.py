import torch
from utils.rotation_utils import superfib_augment_batch

# Here, we will compute
# 1) A trace of the MSE between y and each rotation
# 2) The average rotation error through an entire orbit 

def get_orbit_variance(model, pos, z, y, k):
    device = model.device

    assert model.device == pos.device and model.device == z.device, "Model device doesn't match pos device!"
    assert model.device == z.device, "Model device doesn't match z device!"
    assert model.device == y.device, "Model device doesn't match z device!"

    N = pos.size(0)
    batch = torch.zeros(N, dtype=torch.long, device=device)
    y = torch.zeros(1, device=device)

    pos_aug, z_aug, y_aug, batch_aug = superfib_augment_batch(
        pos, z, y, batch, k, device=device
    )

    # each augmentation corresponds to one molecule
    y_pred = []
    

    for i in range(k):
        mask = (batch_aug == i)
        pos_i = pos_aug[mask]
        z_i   = z_aug[mask]
        batch_i = torch.zeros_like(z_i)

        y_pred.append(model(pos_i, z_i, batch_i))

    y_pred = torch.stack(y_pred)

    mse_per_rotation = (y_pred - y.squeeze())**2 #MSE per rotation
    orbit_variance = y_pred.var(unbiased=False).item() #Variance of predictions
    mean_mse = mse_per_rotation.mean().item()

    return mse_per_rotation, orbit_variance, mean_mse