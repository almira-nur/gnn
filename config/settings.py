import torch

DEVICE = 'best'  # options: 'cpu', 'cuda', 'mps', 'best'
SEED = 42
BATCH_SIZE = 16
LR = 1e-3
HIDDEN_DIM = 128
N_LAYERS = 3

#I'd just keep this true.
SHUFFLE = True

TRAIN_DATA = 'data/mini_1_conf_qm7x_processed_train.h5'
TEST_DATA = 'data/mini_1_conf_qm7x_processed_test.h5'
VAL_DATA = 'data/mini_1_conf_qm7x_processed_val.h5'

# Uncomment for slurm
#TRAIN_DATA = '/home/ptim/orcd/scratch/data/mini_10_conf_qm7x_processed_train.h5'
#TEST_DATA = '/home/ptim/orcd/scratch/data/mini_10_conf_qm7x_processed_test.h5'
#VAL_DATA = '/home/ptim/orcd/scratch/data/mini_10_conf_qm7x_processed_val.h5'

CHECKPOINT_PATH = 'checkpoints'

# Uncomment for slurm
#CHECKPOINT_PATH = '/home/ptim/course_projects/gnn/checkpoints'

NUM_EPOCHS = 50
VERBOSE = False
SHUFFLE = True

N_RBF = 16
CUTOFF = 5.0 # None if no cutoff.

# Max atomic number is 17 (Cl)
EMBEDDING_SIZE = 18

#Global epsilon for clamping minima to avoid division by zero.  Should never need it, and unlikely to need to change it.
EPSILON = 1e-9





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


