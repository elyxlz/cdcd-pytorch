from typing import TypeVar, Optional, Any
from typing_extensions import TypeGuard

import torch
from torch import Tensor
from itertools import repeat
from enum import Enum
import collections.abc
import torch

V = TypeVar("V")


def exists(val: Optional[V]) -> TypeGuard[V]:
    return val is not None

def default(val: Optional[V], d: V) -> V:
    return val if exists(val) else d

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def rand_bool(shape: Any, proba: float, device: Any = None) -> Tensor:
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)