import torch
import numpy as np
import random


def set_seeds():
    random.seed(42)
    np.random.seed(12345)
    torch.manual_seed(1234)
    torch.set_deterministic(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

