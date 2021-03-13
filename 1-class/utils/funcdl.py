import random

import torch
import numpy as np


def set_seed(SEED):
    """Sets random seed everywhere"""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

"""
In main:

set_seed(seed)
"""