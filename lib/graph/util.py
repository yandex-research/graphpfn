import enum

import dgl
import numpy as np
import torch
from torch import Tensor


class Setting(enum.Enum):
    TRANSDUCTIVE = "transductive"
    INDUCTIVE = "inductive"


def pack_tensor(x: Tensor, k: int) -> Tensor:
    size = x.shape[0] // k
    x = torch.stack(torch.split(x, size), dim=1)
    return x


def unpack_tensor(x: Tensor) -> Tensor:
    x = x.transpose(0, 1).contiguous().reshape(-1, *x.shape[2:])
    return x


def pack_array(x: np.ndarray, k: int) -> np.ndarray:
    x = np.stack(np.split(x, k), axis=1)
    return x


def unpack_array(x: np.ndarray) -> np.ndarray:
    x = x.transpose(0, 1).reshape(-1, *x.shape[2:])
    return x


def batch_graph(graph: dgl.DGLGraph, k: int) -> dgl.DGLGraph:
    batched_graph = dgl.batch([graph for _ in range(k)])
    return batched_graph


def batch_tensor(x: Tensor, k: int) -> Tensor:
    batched_x = x.repeat(k, *[1 for _ in range(x.ndim - 1)])
    return batched_x
