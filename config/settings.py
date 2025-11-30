import torch

DEVICE = 'best'  # options: 'cpu', 'cuda', 'mps', 'best'
SEED = 42
BATCH_SIZE = 16
LR = 1e-3
HIDDEN_DIM = 128
N_LAYERS = 3
DATA_PATH = 'data/qm7x_processed.h5'

NUM_EPOCHS = 50
VERBOSE = False
SHUFFLE = True

N_RBF = 16
CUTOFF = 5.0 # None if no cutoff.





def _resolve_device(request: str):
    if request == 'cpu':
        return torch.device('cpu')
    if request == 'mps':
        try:
            return torch.device('mps')
        except Exception:
            return torch.device('cpu')
    if request == 'cuda':
        try:
            return torch.device('cuda')
        except Exception:
            return torch.device('cpu')

    try:
        if torch.cuda.is_available():
            return torch.device('cuda')
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device('mps')
    except Exception:
        pass
    return torch.device('cpu')


DEVICE = _resolve_device(DEVICE)


