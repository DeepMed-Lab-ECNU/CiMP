from yacs.config import CfgNode as CN

_C = CN()
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 2
_C.DATA.PIN_MEMORY = False
_C.DATA.NUM_WORKERS = 4
_C.DATA.MODALITY = 'survival'

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME ='CIMP'
_C.MODEL.STATE = 'finetune'
_C.MODEL.DIM = 256
_C.MODEL.NUM_IN_GROUP = 14
_C.MODEL.NUM_CLASS = 4

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 10
_C.TRAIN.WEIGHT_DECAY = 1e-2
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.PATIENCE = 6
_C.TRAIN.THRESHOLD = 0.95

# -----------------------------------------------------------------------------
# Learning rate settings
# -----------------------------------------------------------------------------
_C.TRAIN.LR_SCHEDULER = CN()
# 'StepLR' or 'Cosine'
_C.TRAIN.LR_SCHEDULER.NAME = 'StepLR'

# Cosine scheduler
_C.TRAIN.WARMUP_EPOCHS = 0
_C.TRAIN.WARMUP_LR = 1e-7
_C.TRAIN.MIN_LR = 5e-6

# StepLR scheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 20
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.9

# -----------------------------------------------------------------------------
# Optimizer settings
# -----------------------------------------------------------------------------
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Misc settings
# -----------------------------------------------------------------------------
_C.CHECKPOINTS_PATH = 'checkpoints'
_C.CHECKPOINTS_NAME = None
# Fixed random seed
_C.SEED = 10241024
_C.DEVICES = 'cuda'


def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()

    return config
