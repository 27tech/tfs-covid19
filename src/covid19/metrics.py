from torch import Tensor
from torch import sqrt
from torch.functional import F
from fastai.torch_core import flatten_check
import torch

__all__ = ['smape', 'mape', 'rmse', 'mpe', 'mape2']


def smape(y_pred: Tensor, target: Tensor) -> Tensor:
    y_pred, target = flatten_check(y_pred, target)
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()


def mape(y_pred, target) -> Tensor:
    y_pred, target = flatten_check(y_pred, target)
    loss = (y_pred - target).abs() / (target.abs() + 1e-8)
    return loss.mean()


def mpe(y_pred, target) -> Tensor:
    y_pred, target = flatten_check(y_pred, target)
    loss = (y_pred - target) / (target + 1e-8)
    return loss.mean()


def rmse(y_pred, target) -> Tensor:
    return sqrt(F.mse_loss(y_pred, target))


def mape2(output, target):
    return torch.mean(torch.abs((target - output) / (target + 1e-8)))
