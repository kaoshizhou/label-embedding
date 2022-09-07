import numpy as np
import torch
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    keys = batch[0].keys()
    res = {key: [] for key in keys}
    [res[key].extend(item[key]) for item in batch for key in item]
    return res

