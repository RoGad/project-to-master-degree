import os
import random
import numpy as np
import torch

from src import config


def set_seed(seed=None):
    seed = config.SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(wm, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(wm.state_dict(), path)


def load_checkpoint(wm, path):
    wm.load_state_dict(torch.load(path, map_location=config.DEVICE))
    return wm
