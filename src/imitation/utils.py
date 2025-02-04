from torch.utils.data._utils.collate import collate, default_collate_fn_map, collate_numpy_array_fn, collate_float_fn
import torch
import numpy as np


def _collate_array(batch, *, collate_fn_map):
    return collate_numpy_array_fn(batch, collate_fn_map=collate_fn_map).to(torch.float32)


def _collate_float(batch, *, collate_fn_map):
    return collate_float_fn(batch, collate_fn_map=collate_fn_map).to(torch.float32)


def collate_to_float32(batch):
    return collate(batch, collate_fn_map=default_collate_fn_map | {np.ndarray: _collate_array, float: _collate_float})
