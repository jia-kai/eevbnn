from .utils import (get_nr_correct, assign_zero_grad, torch_as_npy,
                    get_rng_if_none, iter_prod_range)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod, ABCMeta
from contextlib import contextmanager
import collections
import typing

class AttackLoss(metaclass=ABCMeta):
    @abstractmethod
    def get_loss(self, output: torch.Tensor) -> torch.Tensor:
        """compute loss from given input and output; note that it should not
        perform reduction
        """

    @abstractmethod
    def get_prec(self, output: torch.Tensor) -> typing.Union[int, float]:
        """get a precision measure to be returned as the result"""


class XentAttackLoss(AttackLoss):
    _target = None

    def __init__(self, target: torch.Tensor):
        self._target = target

    def get_loss(self, output):
        return F.cross_entropy(output, self._target, reduction='none')

    def get_prec(self, output):
        return get_nr_correct(output, self._target)


class NllAttackLoss(XentAttackLoss):
    def get_loss(self, output):
        return F.nll_loss(output, self._target, reduction='none')


class CwAttackLoss(XentAttackLoss):
    """max score of wrong classes subtracted by the score of correct class; see
    also ``Towards Evaluating the Robustness of Neural Networks''
    """

    _target_mask = None

    def get_loss(self, output):
        assert output.ndim == 2
        if self._target_mask is None:
            with torch.no_grad():
                mask = torch.zeros_like(output)
                idx = self._target.view(output.size(0), 1)
                mask.scatter_(1, idx, 1)
            self._target_mask = mask

        mask = self._target_mask
        correct_score = (output * mask).sum(dim=1)
        wrong_score = (output * (1 - mask)).max(dim=1).values
        return -F.relu(correct_score - wrong_score + 1e-3)

loss_fn_map = {
    'xent': XentAttackLoss,
    'nll': NllAttackLoss,
    'cw': CwAttackLoss
}


@contextmanager
def model_with_input_grad(model: nn.Module, restore):
    """set model to eval mode and disable gradients for parameters"""
    if restore:
        def restore_fn(orig_req=[(i, i.requires_grad)
                              for i in model.parameters()]):
            for i, j in orig_req:
                i.requires_grad_(j)

    for i in model.parameters():
        i.requires_grad_(False)
    to_training = model.training
    model.eval()

    yield model

    if restore:
        restore_fn()
    if to_training:
        model.train()

def fgsm_batch(model: nn.Module, data: torch.Tensor, loss_fn: AttackLoss,
               epsilon: typing.Union[float, typing.List[float]],
               xmin: float = 0., xmax: float = 1.) -> (
                   typing.Tuple[int, typing.List[typing.Tuple[torch.Tensor, int]
                                                 ]
                                ]
               ):
    """run FGSM (Fast Gradient Sign Attack) on a batch of data on a
    classification model

    :param model: the model to be tested; note that it will be modified: require
        grad of params would be set False

    :return: ``(correct0, [(img, correct1)])`` where ``img`` is the batch of
        adversarial images, ``correct0`` is the number of correct precitions on
        original data, and ``correct1`` is the number of correct predictions
        after the attack. An ``(img, correct1)`` pair is returned for each
        ``epsilon`` value.
    """

    assert data.min().item() >= xmin and data.max().item() <= xmax
    if not isinstance(epsilon, collections.Iterable):
        epsilon = [epsilon]
    epsilon = list(map(float, epsilon))

    data.requires_grad_(True)
    for i in model.parameters():
        i.requires_grad_(False)
    model.eval()
    assign_zero_grad(data)
    output = model(data)
    loss = loss_fn.get_loss(output).sum()
    loss.backward()
    data_grad_sgn = data.grad.sign()

    ret = []
    for eps in epsilon:
        adv_data = torch.clamp(data + data_grad_sgn * eps, xmin, xmax)
        ret.append((adv_data, loss_fn.get_prec(model(adv_data))))
    return loss_fn.get_prec(output), ret


def pgd_batch(model: nn.Module, data: torch.Tensor, loss_fn: AttackLoss,
              epsilon: float, nr_step: int = 10, *,
              step_size_factor: float = 1.5, step_size_factor_min: float = 0.3,
              step_size: float = None,
              xmin: float = 0., xmax: float = 1., restore=False,
              adv_init: torch.Tensor=None,
              dbg_known_adv=None) -> typing.Tuple[torch.Tensor, int]:
    """run PGD (Projected Gradient Descent) on a batch of data on a
    classification model, with random initialization

    :param model: the model to be tested; note that it will be modified:
        required grad of params would be set False

    :param epsilon: max deviation from input value
    :param step_size: if not provided, it would be computed as
        ``max(min(step_size_factor / nr_step, 1), step_size_factor_min) *
        epsilon``
    :param restore: whether to restore model before returning

    :param adv_init: initial point to compute adversarial examples

    :param dbg_known_adv: know adversarial input, for debugging this PGD process

    :return: ``(img, nr_correct)`` where ``img`` is the batch of adversarial
        images, and ``get_nr_correct`` is the number of correct predictions
        after the attack
    """

    assert data.min().item() >= xmin and data.max().item() <= xmax

    data_range = ((data - epsilon).clamp_(xmin, xmax),
                  (data + epsilon).clamp_(xmin, xmax))

    def clip_inplace(x):
        x = torch.min(x, data_range[1], out=x)
        x = torch.max(x, data_range[0], out=x)
        return x

    if adv_init is None:
        adv_init = torch.empty_like(data).uniform_(-epsilon, epsilon).add_(data)
    else:
        assert adv_init.shape == data.shape
        adv_init = adv_init.clone()

    data = clip_inplace(adv_init)

    if step_size is None:
        step_size = max(min(step_size_factor / nr_step, 1),
                        step_size_factor_min) * epsilon
    step_size = float(step_size)

    mask_shape = list(data.shape)
    mask_shape[1:] = [1] * (len(mask_shape) - 1)
    with model_with_input_grad(model, restore) as model:
        best_result = data.clone()
        data.requires_grad_(True)
        last_loss = best_loss = loss_fn.get_loss(model(data))

        for step in range(nr_step):
            last_loss.sum().backward()
            with torch.no_grad():
                grad_sign = data.grad.sign_()
                data = clip_inplace(data.add_(grad_sign.mul_(step_size)))

            data.requires_grad_(False)
            assign_zero_grad(data)
            data.requires_grad_(True)

            last_loss = loss_fn.get_loss(model(data))

            with torch.no_grad():
                mask = last_loss > best_loss
                best_result = clip_inplace(
                    torch.where(mask.view(mask_shape), data, best_result)
                )
                best_loss = torch.where(mask, last_loss, best_loss)

        with torch.no_grad():
            return best_result, loss_fn.get_prec(model(best_result))

def pgd_batch_bin(
    model: nn.Module, data: torch.Tensor, device, input_bits,
    loss_fn: AttackLoss, epsilon: int = 1,
    nr_step: int = 10, step_size_factor: float = 1.5, restore=False,
    xmin: int = 0, xmax: int = 255, channel_dim: int=1,
    rng=None) -> (
        typing.Tuple[torch.Tensor, int]):
    """run PGD attack on a model with binary bit input

    The model should take bit inputs, and ``data`` should provide integer
    values; returned adversarial input contains bit values rather than integer
    values. ``xmin`` and ``xmax`` should be shifted to retain only neeed number
    of bits.

    :param channel_dim: dimension for input channels, along which the bits
        should be unpacked

    :return: ``(adv_bits_input, adv_int_input, accuracy)``
    """

    assert input_bits >= 1 and input_bits <= 8
    assert (type(epsilon) is int and
            epsilon > 0 and
            epsilon < (1 << (input_bits - 1))), (
                f'invalid {epsilon=} {input_bits=}')

    xmin >>=  8 - input_bits
    xmax >>=  8 - input_bits

    orig_dtype = data.dtype
    orig_device = data.device

    data = torch_as_npy(data)
    assert data.dtype == np.uint8
    assert data.min() >= xmin and data.max() <= xmax, (
        f'data_range=[{data.min()}, {data.max()}] {xmin=} {xmax=}')

    data = data.astype(np.int32)

    data_range = (np.clip(data - epsilon, xmin, xmax),
                  np.clip(data + epsilon, xmin, xmax))

    def clip_inplace(x):
        return np.clip(x, data_range[0], data_range[1], out=x)

    rng = get_rng_if_none(rng)

    data = clip_inplace(
        data + rng.randint(-epsilon, epsilon + 1,
                           size=data.shape, dtype=data.dtype))

    # coefficient of each bit to sum the gradients
    grad_coeff = np.zeros(input_bits, dtype=np.float32)
    eps_bits = epsilon.bit_length()
    grad_coeff[-eps_bits:] = 0.5 ** np.arange(eps_bits)

    assert channel_dim == 1 and data.ndim == 4, 'unimplemented config'

    n, c, h, w = data.shape
    grad_shape = (n, c, input_bits, h, w)
    grad_coeff = (torch.from_numpy(grad_coeff).
                  to(device).view(1, 1, input_bits, 1, 1))

    def as_torch_bits_tensor(x: np.ndarray):
        x = x.reshape(n, c, 1, h, w).astype(np.uint8)
        bits = np.unpackbits(x, axis=2)
        bits = bits[:, :, -input_bits:].reshape(n, c * input_bits, h, w)
        ret = torch.from_numpy(bits).to(device).to(torch.float32).detach_()
        ret.requires_grad_(True)
        return ret

    def grad_reduce_bits_sum(x: torch.Tensor):
        return (x.view(grad_shape).mul_(grad_coeff)).sum(dim=channel_dim+1)

    step_size = int(max(1, round(min(step_size_factor / nr_step, 1) * epsilon)))

    with model_with_input_grad(model, restore) as model:
        for i in range(nr_step):
            data_torch = as_torch_bits_tensor(data)
            loss_fn.get_loss(model(data_torch)).backward()
            with torch.no_grad():
                grad = grad_reduce_bits_sum(data_torch.grad)
                step_torch = grad.sign_().to(torch.int32).mul_(step_size)
            data += torch_as_npy(step_torch)
            data = clip_inplace(data)

        data_bits = as_torch_bits_tensor(data)
        data_int = torch.from_numpy(data).to(orig_device, orig_dtype)
        return data_bits, data_int, loss_fn.get_prec(model(data_bits))
