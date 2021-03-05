from contextlib import contextmanager
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import pickle

from collections import OrderedDict

import pytorch_lightning as pl


@contextmanager
def timer(message: str):
    print(f'[{message} start.]')
    t0 = time.time()
    yield
    elapsed_time = time.time() - t0
    print(f'[{message}] done in {elapsed_time / 60:.1f} min.')


def set_seed(seed: int = 2021):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def read_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_model(filepath, device="cpu"):
    model_obj = torch.load(filepath, map_location=device)
    baseline = list(model_obj['callbacks'].values())[0]['best_model_score'].item()
    optimizer_state = model_obj['optimizer_states']
    model_state = model_obj['state_dict']
    return baseline, optimizer_state, model_state


def remove_prefix_from_dict(dictionary, prefix):
    new_dict = OrderedDict()
    for k, v in dictionary.items():
        name = k[len(prefix):]
        new_dict[name] = v
    return new_dict


def dict_to_device(dictionary, device):
    for k, v in dictionary.items():
        dictionary[k] = v.to(device)
    return dictionary


def prob_topk(input, topk):
    values, indices = torch.topk(input, topk)
    probs = torch.zeros(3, 5).scatter_(1, indices, values)
    probs = torch.softmax(probs, dim=-1)
    return probs


if __name__ == "__main__":
    import config
    model_path = config.PRETRAINED_MODEL
    load_model(filepath=model_path, device="cpu")