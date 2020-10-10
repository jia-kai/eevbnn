import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils

import itertools
import threading
import queue
import os
import typing
import contextlib
from pathlib import Path

def get_rng_if_none(rng, seed: int=42):
    """get a predefined stable rng if rng is None"""
    if rng is None:
        return np.random.RandomState(seed)
    assert isinstance(rng, np.random.RandomState)
    return rng

def iter_prod_range(*sizes):
    """iterate over product of ranges of given sizes"""
    return itertools.product(*map(range, sizes))

class ModelHelper:
    """a mixin class to add basic functionality to torch modules

    Note: args for the network constructor should be passed via ``kwargs`` in
     :meth:`create_with_load`, so it can be saved in the dict.
    """

    __ctor_args = None
    _ctor_args_key = '__ModelHelper_ctor_args'

    def save_to_file(self, fpath, state_only=False):
        if isinstance(fpath, Path):
            fpath = str(fpath)
        if state_only:
            state = self.state_dict()
            state[self._ctor_args_key] = self.__ctor_args
        else:
            state = self
        torch.save(state, fpath)

    @classmethod
    def create_with_load(cls, fpath=None, kwargs={}, enforce_state_load=False):
        """note that ``fpath`` can either contain a full model or only the state
        dict

        :param enforce_state_load: if the complete model class is saved, whether
            to load only the states and assign to this class, or return the
            loaded class directly
        """
        if fpath is not None:
            if isinstance(fpath, Path):
                fpath = str(fpath)
            state = torch.load(fpath, map_location=torch.device('cpu'))
            if isinstance(state, ModelHelper):
                if enforce_state_load:
                    state = state.state_dict()
                else:
                    state._on_load_from_file()
                    return state
            kwargs = kwargs.copy()
            kwargs.update(state.pop(cls._ctor_args_key, {}))
            ret = cls(**kwargs)
            ret.load_state_dict(state)
            return ret
        else:
            ret = cls(**kwargs)
            ret.__ctor_args = kwargs
            w = getattr(ret, 'weights_init', None)
            if w is not None:
                ret.apply(w)
                print('custom weight init used for {}'.format(ret))
            return ret

    def _on_load_from_file(self):
        """can be overriden by subclasses to get notification when model is
        loaded from file"""


def torch_as_npy(x: torch.Tensor) -> np.ndarray:
    return x.data.cpu().numpy()

def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def make_grid_to_cv2uint8(tensor, *args, **kwargs):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    ret = torch_as_npy(vutils.make_grid(tensor, *args, **kwargs))
    ret = ret.transpose(1, 2, 0)
    return np.clip(0, 255, ret * 255).astype(np.uint8)

def npy_chw_f32_to_cv2_uint8(img: np.ndarray):
    """convert a ``(C, H, W)`` float32 tensor in [0, 1] into cv2 uint8 format"""
    if img.ndim == 2:
        img = img[np.newaxis]
    assert img.ndim == 3 and img.shape[0] in [1, 3]
    return np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)

def get_nr_correct(outputs, labels):
    """get number of correct predictions for classification problems, where
    prediction is computed from argmax"""
    pred_label = torch.argmax(outputs, dim=1)
    return int(torch.eq(pred_label, labels).to(torch.int32).sum().item())

def assign_zero_grad(x: torch.Tensor):
    """set gradient associated with a tensor to zero; similar to
    :meth:`torch.nn.Module.zero_grad`

    :return: x
    """
    if x.grad is not None:
        x.grad.detach_().zero_()
    return x

def assign_param(x: torch.nn.Parameter,
                 val: typing.Union[torch.Tensor, np.ndarray]):
    """modify the value of a torch parameter"""
    rg = x.requires_grad
    if isinstance(val, np.ndarray):
        val = torch.from_numpy(val)
    assert x.shape == val.shape, (x.shape, val.shape)
    try:
        x.requires_grad = False
        x.copy_(val)
    finally:
        x.requires_grad = rg
    return x

class PrintLayerSizes:
    def register_to(self, network: torch.nn.Module):
        network.register_forward_pre_hook(self)
        for i in network.modules():
            i.register_forward_pre_hook(self)

    def __call__(self, module, inp):
        print('{}: {}'.format(module.__class__.__name__,
                              [i.size() for i in inp]))

class PrintNorm:
    def register_to(self, network: torch.nn.Module):
        network.register_forward_hook(self)

    def __call__(self, module, inp, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        print('Inside ' + module.__class__.__name__ + ' forward')
        print('')
        print('input: ', type(inp))
        print('input[0]: ', type(inp[0]))
        print('output: ', type(output))
        print('')
        print('input size:', inp[0].size())
        print('output size:', output.data.size())
        print('input range:', inp[0].min(), inp[0].max())
        print('output norm:', output.data.norm())


class ModelWithNormalize(nn.Module):
    """a model that combines an existing model with input normalization"""
    normalize = None
    _saved_result = None

    def __init__(self, model: nn.Module, normalize: transforms.Normalize):
        super().__init__()
        self.model = model
        self.normalize = normalize
        self._saved_result = []

    def forward(self, x):
        if not self._saved_result:
            mean = torch.as_tensor(self.normalize.mean, dtype=x.dtype,
                                   device=x.device).view(1, -1, 1, 1)
            std = torch.as_tensor(self.normalize.std, dtype=x.dtype,
                                   device=x.device).view(1, -1, 1, 1)
            mean.requires_grad = False
            std.requires_grad = False
            self._saved_result += [mean, std]
        mean, std = self._saved_result
        return self.model.forward((x - mean) / std)


def default_dataset_root() -> str:
    """default dataset root directory"""
    return str(Path(__file__).resolve().parent / 'data')

class LimitedDatasetIter:
    """iterate over a pytorch dataset with a limit on the number of iterations
    per epoch

    The iterator would stop if either 1) the underlying iterator stops 2)
    specified limit is reached. In the second case, the next iteration would
    continue from previous iteration.
    """
    _parent_iter = None
    _parent_iobj = None
    _limit = None

    def __init__(self, parent_iter: typing.Iterable, limit: int):
        self._parent_iter = parent_iter
        self._limit = limit

    def __iter__(self):
        if self._parent_iobj is None:
            self._parent_iobj = iter(self._parent_iter)
        try:
            for i in range(self._limit):
                yield next(self._parent_iobj)
        except StopIteration:
            del self._parent_iobj


class ReplayDatasetIter:
    """reuse each item for given number of times before going to the next one"""
    _replay = None
    _parent_iter = None

    def __init__(self, parent_iter: typing.Iterable, replay: int):
        self._parent_iter = parent_iter
        self._replay = replay

    def __iter__(self):
        for data in self._parent_iter:
            for i in range(self._replay):
                yield data


class AverageMetric:
    _last = None
    _avg = 0
    _nr = 0

    def add(self, x: float):
        """add a new data point
        :return: self
        """
        x = float(x)
        self._last = x
        self._nr += 1
        k = 1 / self._nr
        self._avg = x * k + self._avg * (1 - k)

    @property
    def avg(self):
        """current average vaue"""
        return self._avg

    @property
    def last(self):
        """last vaue"""
        return self._last


class Flatten(nn.Module):
    def infer_out_shape(self, inp_shape):
        n, *other = inp_shape
        return n, int(np.prod(other))

    def forward(self, x):
        return x.view(x.size(0), -1)


@contextlib.contextmanager
def setup_pyx_import():
    import pyximport
    px = pyximport.install()
    yield
    pyximport.uninstall(*px)


def sqr_hinge_loss(score: torch.Tensor, target: torch.Tensor):
    """squared hinge loss for multi-class classification"""
    assert score.ndim == 2 and target.ndim == 1
    with torch.no_grad():
        mask = torch.ones_like(score)
        idx = target.view(mask.size(0), 1)
        mask = mask.scatter_(1, idx, -1)
    return ((1 + score * mask).relu()**2).mean()

@contextlib.contextmanager
def ensure_training_state(net: nn.Module, train: bool):
    """a context manager to ensure that net.train()/net.eval() state within the
    context, and restore it to the original state"""
    orig_state = net.training
    if train:
        net.train()
    else:
        net.eval()

    try:
        yield
    finally:
        if orig_state:
            net.train()
        else:
            net.eval()
