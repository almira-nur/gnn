import torch
import os

DEVICE = 'cuda'  # options: 'cpu', 'cuda', 'mps', 'best'
SEED = 42
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4 #L2 regularization
HIDDEN_DIM = 64
N_LAYERS = 3
MODEL_TYPE = 'strawberry'  # options: 'vanilla', 'chocolate', 'strawberry'
AUGMENT_TYPE = 'superfib_intermediate'  # Options: 'none', 'superfib_end', 'superfib_intermediate'.  Type of data augmentation to use during training.
N_ROTATIONS_EVALUATION = 10  # Number of rotations to average over during evaluation.
N_ROTATIONS_TRAIN = 4 #0 means no augmentation
LAMBDA_CONSISTENCY = 0.1  # Weight for equivariance layerwise consistency loss term during training when using data augmentation.

RESUME_PATH = None  # Path to checkpoint to resume training from, or None to start fresh.
#RESUME_PATH = "/home/ptim/course_projects/gnn/checkpoints/checkpoint_epoch_18.pt"
#I'd just keep this true.
SHUFFLE = True


FIG_PATH = 'figures'
# Uncomment for slurm
#FIG_PATH = '/home/ptim/course_projects/gnn/figures'

TRAIN_DATA = 'data/mini_200_conf_qm7x_processed_train.h5'
TEST_DATA = 'data/mini_200_conf_qm7x_processed_test.h5'
VAL_DATA = 'data/mini_200_conf_qm7x_processed_val.h5'

# Uncomment for slurm
TRAIN_DATA = '/home/ptim/orcd/scratch/data/mini_200_conf_qm7x_processed_train.h5'
TEST_DATA = '/home/ptim/orcd/scratch/data/mini_200_conf_qm7x_processed_test.h5'
VAL_DATA = '/home/ptim/orcd/scratch/data/mini_200_conf_qm7x_processed_val.h5'


CHECKPOINT_PATH = 'checkpoints'

CHECKPOINT_PATH = '/home/ptim/course_projects/gnn/checkpoints'

NUM_EPOCHS = 20
VERBOSE = False
SHUFFLE = True

N_RBF = 16
CUTOFF = 5.0 # None if no cutoff.

# Max atomic number is 17 (Cl)
EMBEDDING_SIZE = 18

#Global epsilon for clamping minima to avoid division by zero.  Should never need it, and unlikely to need to change it.
EPSILON = 1e-9


RUN_NAME = TRAIN_DATA.rsplit("/", 1)[-1].rsplit(".", 1)[0] + f"_{MODEL_TYPE}_{AUGMENT_TYPE}_hd{HIDDEN_DIM}_nl{N_LAYERS}_bs{BATCH_SIZE}_lr{LR}"

os.makedirs(f'/home/ptim/course_projects/gnn/{RUN_NAME}', exist_ok=True)
os.makedirs(f'/home/ptim/course_projects/gnn/{RUN_NAME}/checkpoints', exist_ok=True)
os.makedirs(f'/home/ptim/course_projects/gnn/{RUN_NAME}/figures', exist_ok=True)
FIG_PATH = f'/home/ptim/course_projects/gnn/{RUN_NAME}/figures'
CHECKPOINT_PATH = f'/home/ptim/course_projects/gnn/{RUN_NAME}/checkpoints'

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
