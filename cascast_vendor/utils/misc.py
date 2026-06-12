# Stripped subset of the original CasCast utils/misc.py.
# Removes megatron_utils, distributed training, and logging helpers that are
# not needed for single-GPU / CPU inference in the course environment.
import torch
import torch.distributed as dist
from typing import Any
import numpy as np


class Dict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def collate_fn(batch):
    batch = list(zip(*batch))
    array_seq = np.stack(batch[0])
    origin_array_seq = np.stack(batch[1])
    date_time_seq = np.stack(batch[2])
    return tuple([array_seq, origin_array_seq, date_time_seq])
