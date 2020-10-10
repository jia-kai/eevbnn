from .utils import ModelHelper, Flatten, ensure_training_state
from .eval_bin import BinConv2dEval, BinLinearEval

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import weakref
import functools

g_weight_decay = 1e-7
g_channel_scale = 1
g_bingrad_soft_tanh_scale = 1
g_weight_mask_std = 0.01
g_use_scalar_scale_last_layer = True
g_remove_last_bn = False

def scale_channels(x: int):
    return max(int(round(x * g_channel_scale)), 1)

class AbstractTensor:
    """tensor object for abstract interpretation (here we use interval
    arithmetic)
    """
    __slots__ = ['vmin', 'vmax', 'loss']

    loss_layer_decay = 1
    """decay of loss of previous layer compared to current layer"""

    def __init__(self, vmin: torch.Tensor, vmax: torch.Tensor,
                 loss: torch.Tensor):
        assert vmin.shape == vmax.shape and loss.numel() == 1
        self.vmin = vmin
        self.vmax = vmax
        self.loss = loss

    def apply_linear(self, w, func):
        """apply a linear function ``func(self, w)`` by decomposing ``w`` into
        positive and negative parts"""
        wpos = F.relu(w)
        wneg = w - wpos
        vmin_new = func(self.vmin, wpos) + func(self.vmax, wneg)
        vmax_new = func(self.vmax, wpos) + func(self.vmin, wneg)
        return AbstractTensor(torch.min(vmin_new, vmax_new),
                              torch.max(vmin_new, vmax_new),
                              self.loss)

    def apply_elemwise_mono(self, func):
        """apply a non-decreasing monotonic function on the values"""
        return AbstractTensor(func(self.vmin), func(self.vmax), self.loss)

    @property
    def ndim(self):
        return self.vmin.ndim

    @property
    def shape(self):
        return self.vmin.shape

    def size(self, dim):
        return self.vmin.size(dim)

    def view(self, *shape):
        return AbstractTensor(self.vmin.view(shape), self.vmax.view(shape),
                              self.loss)


class MultiSampleTensor:
    """tensor object that contains multiple samples within the perturbation
    bound near the natural image

    :var data: a tensor ``(K * N, C, H, W)`` where ``data[0:N]`` is the natural
        image, and data[N::N] are perturbations
    """
    __slots__ = ['k', 'data', 'loss']

    loss_layer_decay = 1
    """decay of loss of previous layer compared to current layer"""

    def __init__(self, k: int, data: torch.Tensor, loss: torch.Tensor = 0):
        self.k = k
        self.data = data
        self.loss = loss
        assert data.shape[0] % k == 0

    @classmethod
    def from_squeeze(cls, data: torch.Tensor, loss=0):
        """``data`` should be in ``(K, N, ...)`` format"""
        k, n, *other = data.shape
        return cls(k, data.view(k * n, *other), loss)

    def apply_batch(self, func, loss=None):
        """apply a batch function"""
        if loss is None:
            loss = self.loss
        return MultiSampleTensor(self.k, func(self.data), loss)

    def as_expanded_tensor(self) -> torch.Tensor:
        """expand the first dimension to ``(K, N)``"""
        kn, *other = self.shape
        k = self.k
        n = kn // k
        return self.data.view(k, n, *other)

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    def size(self, dim):
        return self.data.size(dim)

    def view(self, *shape):
        assert shape[0] == self.shape[0]
        return MultiSampleTensor(self.k, self.data.view(shape), self.loss)


class Binarize01Act(nn.Module):
    class Fn(Function):
        @staticmethod
        def forward(ctx, inp, scale=None):
            """:param scale: scale for gradient computing"""
            if scale is None:
                ctx.save_for_backward(inp)
            else:
                ctx.save_for_backward(inp, scale)
            return (inp >= 0).to(inp.dtype)

        @staticmethod
        def backward(ctx, g_out):
            if len(ctx.saved_tensors) == 2:
                inp, scale = ctx.saved_tensors
            else:
                inp, = ctx.saved_tensors
                scale = 1

            if g_bingrad_soft_tanh_scale is not None:
                scale = scale * g_bingrad_soft_tanh_scale
                tanh = torch.tanh_(inp * scale)
                return (1 - tanh.mul_(tanh)).mul_(g_out), None

            # grad as sign(hardtanh(x))
            g_self = (inp.abs() <= 1).to(g_out.dtype)
            return g_self.mul_(g_out), None

    def __init__(self, grad_scale=1):
        super().__init__()
        self.register_buffer(
            'grad_scale',
            torch.tensor(float(grad_scale), dtype=torch.float32))

    def forward(self, x):
        grad_scale = getattr(self, 'grad_scale', None)
        f = lambda x: self.Fn.apply(x, grad_scale)

        def rsloss(x, y):
            return (1 - torch.tanh(1 + x * y)).sum()

        if type(x) is AbstractTensor:
            loss = rsloss(x.vmin, x.vmax)
            loss += x.loss * AbstractTensor.loss_layer_decay
            vmin = f(x.vmin)
            vmax = f(x.vmax)
            return AbstractTensor(vmin, vmax, loss)
        elif type(x) is MultiSampleTensor:
            rv = x.as_expanded_tensor()
            loss = rsloss(rv[-1], rv[-2])
            return x.apply_batch(
                f,
                loss=x.loss * MultiSampleTensor.loss_layer_decay + loss
            )
        else:
            return f(x)


class Binarize01WeightNoScaleFn(Function):
    @staticmethod
    def forward(ctx, inp):
        v = (inp >= 0).to(inp.dtype)
        ctx.save_for_backward(v)
        return v

    @staticmethod
    def backward(ctx, g_out):
        out, = ctx.saved_tensors
        return (out * g_weight_decay).add_(g_out)


class TernaryWeightFn(Function):
    @staticmethod
    def forward(ctx, inp):
        v = inp.sign().mul_((inp.abs() >= 0.005).to(inp.dtype))
        ctx.save_for_backward(v)
        return v

    @staticmethod
    def backward(ctx, g_out):
        out, = ctx.saved_tensors
        return (out * g_weight_decay).add_(g_out)


class TernaryWeightWithMaskFn(Function):
    """BinMask in the paper; see :func:`binarize_weights`"""
    @staticmethod
    def forward(ctx, inp):
        return inp.sign()

    @staticmethod
    def backward(ctx, g_out):
        return g_out


class IdentityWeightFn(Function):
    """using the original floating point weight"""
    @staticmethod
    def forward(ctx, inp):
        return inp

    @staticmethod
    def backward(ctx, g_out):
        return g_out


class Quant3WeightFn(Function):
    """quantized weight with values in the range [-3, 3]"""

    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        qmin = -0.016
        qmax = 0.016
        # 90 interval of 0.01 normal distribution
        step = (qmax - qmin) / 7
        return torch.clamp((inp - qmin).div_(step).floor_().sub_(3), -3, 3)

    @staticmethod
    def backward(ctx, g_out):
        return g_out

class Quant3WeightWithMaskFn(Quant3WeightFn):
    pass


g_weight_binarizer = TernaryWeightWithMaskFn

def binarize_weights(layer: nn.Module):
    weight: torch.Tensor = layer.weight
    wb = layer.weight_binarizer
    if wb is TernaryWeightWithMaskFn or wb is Quant3WeightWithMaskFn:
        mask = layer._parameters.get('weight_mask')
        if mask is None:
            layer.register_parameter(
                'weight_mask', nn.Parameter(
                    torch.empty_like(weight).
                    normal_(std=g_weight_mask_std).
                    abs_().
                    detach_()
                ))
            mask = layer._parameters['weight_mask']
        return wb.apply(weight) * Binarize01WeightNoScaleFn.apply(mask)

    assert wb in [TernaryWeightFn, IdentityWeightFn, Quant3WeightFn]
    return wb.apply(weight)


class BinConv2d(nn.Conv2d):
    """conv with binarized weights; no bias is allowed"""

    rounding = False

    class RoundFn(Function):
        """apply rounding to compensate computation errors in float conv"""
        @staticmethod
        def forward(ctx, inp):
            return torch.round_(inp)

        @staticmethod
        def backward(ctx, g_out):
            return g_out

        @classmethod
        def g_apply(cls, inp):
            """general apply with handling of :class:`AbstractTensor`"""
            f = cls.apply
            if type(inp) is AbstractTensor:
                return inp.apply_elemwise_mono(f)
            if type(inp) is MultiSampleTensor:
                return inp.apply_batch(f)
            return f(inp)


    def __init__(self, weight_binarizer, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, rounding=True):
        """:param rounding: whether the output should be rounded to integer to
            compensate float computing errors. It should be set when input is
            guaranteed to be int.
        """
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=False)
        self.weight_binarizer = weight_binarizer
        binarize_weights(self)  # create weight_mask
        self.rounding = rounding

    def _do_forward(self, x, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight_bin

        def do_conv(x, w):
            return F.conv2d(x, w, bias, self.stride,
                            self.padding, self.dilation, self.groups)

        if type(x) is AbstractTensor:
            return x.apply_linear(weight, do_conv)

        if type(x) is MultiSampleTensor:
            return x.apply_batch(lambda d: do_conv(d, weight))

        return do_conv(x, weight)

    def forward(self, x):
        y = self._do_forward(x)
        if self.rounding:
            y = self.RoundFn.g_apply(y)
        return y

    @property
    def weight_bin(self):
        return binarize_weights(self)

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.normal_(std=0.01)

    def __repr__(self):
        kh, kw = self.weight.shape[2:]
        return (
            f'{type(self).__name__}({self.in_channels}, {self.out_channels}, '
            f'bin={self.weight_binarizer.__name__}, '
            f'kern=({kh}, {kw}), stride={self.stride}, padding={self.padding})'
        )


class BinLinear(nn.Linear):
    rounding = False

    RoundFn = BinConv2d.RoundFn

    def __init__(self, weight_binarizer, in_features, out_features,
                 rounding=True):
        super().__init__(in_features, out_features, bias=False)
        self.weight_binarizer = weight_binarizer
        binarize_weights(self)  # create weight_mask
        self.rounding = rounding

    def _do_forward(self, x, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight_bin

        matmul = functools.partial(F.linear, bias=bias)

        if type(x) is AbstractTensor:
            return x.apply_linear(weight, matmul)

        if type(x) is MultiSampleTensor:
            return x.apply_batch(lambda d: matmul(d, weight))

        return matmul(x, weight)

    def forward(self, x):
        y = self._do_forward(x)
        if self.rounding:
            y = self.RoundFn.g_apply(y)
        return y

    @property
    def weight_bin(self):
        return binarize_weights(self)

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.normal_(std=0.01)


class PositiveInputCombination:
    @classmethod
    def bias_from_bin_weight(cls, weight):
        return F.relu_(-weight.view(weight.size(0), -1)).sum(dim=1)

    def get_bias(self):
        """equivalent to ``self.bias_from_bin_weight(self.weight_bin)``"""
        return self.bias_from_bin_weight(self.weight_bin)


class BinConv2dPos(BinConv2d, PositiveInputCombination):
    """binarized conv2d where the output is always positive (i.e. treat -1
    weight as adding the bool negation of input var)"""

    def _do_forward(self, x):
        weight_bin = self.weight_bin
        return super()._do_forward(
            x, weight=weight_bin,
            bias=self.bias_from_bin_weight(weight_bin))


class BinLinearPos(BinLinear, PositiveInputCombination):
    def _do_forward(self, x):
        weight_bin = self.weight_bin
        return super()._do_forward(
            x, weight=weight_bin,
            bias=self.bias_from_bin_weight(weight_bin))


class ScaleBias(nn.Module):
    """scale features and add bias for classfication"""

    def __init__(self, nr_classes):
        super().__init__()
        self.nr_classes = nr_classes
        self.scale = nn.Parameter(
            torch.from_numpy(np.array(1 / nr_classes, dtype=np.float32)))
        self.bias = nn.Parameter(
            torch.from_numpy(np.zeros(nr_classes, dtype=np.float32))
        )

    def forward(self, x):
        return self.scale * x + self.bias.view(-1, self.nr_classes)

    def get_scale_bias_eval(self):
        return self.scale, self.bias


class BatchNormStatsCallbak(nn.Module):
    """batchnorm with callback for scale and bias:
    ``owner.on_bn_internals(bn, scale, bias)`` would be called each time
    :meth:`forward` gets executed
    """
    use_scalar_scale = False
    bias_regularizer_coeff = 1

    def __init__(self, owner, dim, momentum=0.9, eps=1e-5,
                 use_scalar_scale=False):
        self.fix_ownership(owner)

        super().__init__()
        self.momentum = momentum
        self.eps = eps
        dim_scale = 1 if use_scalar_scale else dim
        self.register_buffer(
            'running_var', torch.zeros(dim_scale, dtype=torch.float32))
        self.register_buffer(
            'running_mean', torch.zeros(dim, dtype=torch.float32))
        self.weight = nn.Parameter(torch.ones(dim_scale, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.use_scalar_scale = use_scalar_scale

    def fix_ownership(self, owner):
        # bypass pytorch hooks
        self.__dict__['owner'] = weakref.proxy(owner)

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'{self.output_dim}, scalar={self.use_scalar_scale}, '
                f'cbd={self.bias_regularizer_coeff})')

    def forward(self, inp: torch.Tensor):
        if type(inp) is MultiSampleTensor:
            return inp.apply_batch(self.forward)

        if inp.ndim == 2:
            reduce_axes = [0]
            def view(x):
                return x.view(1, x.size(0))
        else:
            assert inp.ndim == 4
            reduce_axes = [0, 2, 3]
            def view(x):
                return x.view(1, x.size(0), 1, 1)

        if type(inp) is AbstractTensor:
            scale, bias = self._get_scale_bias(
                self.running_var, self.running_mean)
            scale = view(scale)
            bias = view(bias)
            return inp.apply_linear(scale, lambda x, w: x * w + bias)

        if self.training or (not self.training and self.owner.eval_with_bn):
            if self.use_scalar_scale:
                var = inp.flatten().var(unbiased=True)
            else:
                var = inp.var(reduce_axes, unbiased=True)
            mean = inp.mean(reduce_axes)
            if not self.owner.eval_with_bn:
                with torch.no_grad():
                    (self.running_var.
                     mul_(self.momentum).
                     add_(var * (1 - self.momentum)))
                    (self.running_mean.
                     mul_(self.momentum).
                     add_(mean * (1 - self.momentum)))
        else:
            var = self.running_var
            mean = self.running_mean

        scale, bias = self._get_scale_bias(var, mean)
        ret = inp * view(scale) + view(bias)
        self.owner.on_bn_internals(self, scale, bias)
        return ret

    def _get_scale_bias(self, var, mean):
        std = torch.sqrt(var + self.eps)
        scale = self.weight / std
        bias = self.bias - mean * scale
        return scale, bias

    def get_scale_bias_eval(self):
        """get scale and bias using computed statistics"""
        return self._get_scale_bias(self.running_var, self.running_mean)

    @property
    def temperature(self):
        assert self.use_scalar_scale
        return self.weight.item()

    @property
    def output_dim(self):
        return self.running_mean.size(0)


def cvt_to_eval_sequential(model: nn.Sequential) -> nn.Sequential:
    """convert a model to binary model in eval mode, using layers from
    :mod:`.eval_bin`

    :return: a model with attributes: ``quant_step`` and ``is_mlp``
    """
    idx = 0
    result = []

    def add_linear_bn_act(layer, idx, cvt):
        bn = model[idx + 1]
        act = model[idx + 2]
        assert isinstance(bn, (nn.BatchNorm2d, nn.BatchNorm1d,
                               BatchNormStatsCallbak)), (
            f'unexpected next layer {bn}')
        assert isinstance(act, Binarize01Act), (
            f'unexpected activation {act}')
        result.append(cvt(layer, bn))
        return idx + 3

    quant_step = None
    is_mlp = False
    while idx < len(model):
        layer = model[idx]
        if isinstance(layer, InputQuantizer):
            assert idx == 0
            assert quant_step is None
            quant_step = layer.step
            conv = model[idx + 1]

            if isinstance(conv, Flatten):
                # MLP model, starting with inp-quant, flatten, linear, bn, act
                result.append(Flatten())
                idx += 2
                lin = model[idx]
                assert isinstance(lin, BinLinear), (
                    f'unexpected layer after Flatten: {lin}')
                idx = add_linear_bn_act(
                    lin, idx,
                    functools.partial(BinLinearEval.from_lin_bn,
                                      quant_step=layer.step))
                is_mlp = True
                continue

            assert isinstance(conv, BinConv2d), (
                f'unexpected layer following InputQuantizer: {conv}')
            idx = add_linear_bn_act(
                conv, idx + 1,
                functools.partial(BinConv2dEval.from_conv_bn,
                                  quant_step=layer.step))
            continue

        if isinstance(layer, BinConv2d):
            idx = add_linear_bn_act(layer, idx, BinConv2dEval.from_conv_bn)
            continue
        if isinstance(layer, Flatten):
            result.append(Flatten())
            idx += 1
            continue
        if isinstance(layer, BinLinear):
            if idx + 2 < len(model):
                idx = add_linear_bn_act(layer, idx, BinLinearEval.from_lin_bn)
                continue

            # now idx >= len(model) - 2
            if idx + 1 < len(model):
                scale_layer = model[idx + 1]
                assert isinstance(scale_layer, (ScaleBias,
                                                BatchNormStatsCallbak))
                scale, bias = scale_layer.get_scale_bias_eval()
                if isinstance(layer, PositiveInputCombination):
                    bias = scale * layer.get_bias() + bias
                scale_bias = scale, bias
            else:
                scale_bias = None
            result.append(BinLinearEval(layer.weight_bin, None, scale_bias,
                                        layer.rounding))
            break

        raise RuntimeError(f'unsupported layer: {layer}')

    ret = nn.Sequential(*result)
    ret.quant_step = quant_step
    ret.is_mlp = is_mlp
    return ret


def setattr_inplace(obj, k, v):
    assert hasattr(obj, k)
    setattr(obj, k, v)
    return obj

class SeqBinModelHelper:
    eval_with_bn = False
    """perform batch normalization even in eval mode: normalize the minibatch
    without updating running mean/var stats"""

    def eval_with_bn_(self, flag: bool):
        """inplace setter of the :attr:`eval_with_bn` flag"""
        self.eval_with_bn = bool(flag)
        return self

    def get_bn_state(self):
        """copy BN states into a list"""
        ret = []
        for i in self.features:
            if isinstance(i, BatchNormStatsCallbak):
                ret.append((
                    i.momentum, i.running_mean.clone(), i.running_var.clone()
                ))
        return ret

    def restore_bn_state(self, state):
        """restore states of BN from a list; return self"""
        idx = 0
        for i in self.features:
            if isinstance(i, BatchNormStatsCallbak):
                mom, mean, var = state[idx]
                idx += 1
                i.momentum = mom
                i.running_mean.copy_(mean)
                i.running_var.copy_(var)
        assert idx == len(state)
        return self

    def forward(self, x):
        if type(x) is AbstractTensor:
            assert isinstance(self.features[-3], Binarize01Act)
            return self.features[:-2](x)

        return self.features(x)

    def forward_with_multi_sample(
            self, x: torch.Tensor, x_adv: torch.Tensor,
            eps: float,
            inputs_min: float = 0, inputs_max: float = 1):
        """forward with randomly sampled perturbations and compute a
        stabilization loss """
        data = [x_adv, None, None]
        eps = float(eps)
        with torch.no_grad():
            delta = torch.empty_like(x).random_(0, 2).mul_(2*eps).sub_(eps)
            data[1] = torch.clamp_min_(x - delta, inputs_min)
            data[2] = torch.clamp_max_(x + delta, inputs_max)
            data = torch.cat([i[np.newaxis] for i in data], dim=0)
        y = self.forward(MultiSampleTensor.from_squeeze(data))
        return y.as_expanded_tensor()[0], y.loss

    def compute_act_stabilizing_loss_abstract(
            self, inputs: torch.Tensor, eps: float,
            inputs_min: float = 0, inputs_max: float = 1):
        """compute an extra loss for stabilizing the activations using abstract
            interpretation

        :return: loss value
        """
        loss = torch.tensor(0, dtype=torch.float32, device=inputs.device)
        with torch.no_grad():
            imin = torch.clamp_min_(inputs - eps, inputs_min)
            imax = torch.clamp_max_(inputs + eps, inputs_max)
        return self.forward(AbstractTensor(imin, imax, loss)).loss

    def cvt_to_eval(self):
        return cvt_to_eval_sequential(self.features)

    @property
    def temperature(self):
        return self.features[-1].temperature

    def on_bn_internals(self, bn, scale, bias):
        pass

    def get_sparsity_stat(self):
        """:return: list of layer sparsity, total number of zeros, total number
        of weights"""
        nr_zero = 0
        tot = 0
        parts = []
        for i in self.features:
            if isinstance(i, (BinConv2d, BinLinear)):
                with torch.no_grad():
                    wb = i.weight_bin
                    nz = int((wb.abs() < 1e-4).to(dtype=torch.int32).sum())
                    cur_nr = wb.numel()
                tot += cur_nr
                nr_zero += nz
                parts.append(nz / cur_nr)
        return parts, nr_zero, tot

    def _on_load_from_file(self):
        for i in self.features:
            if isinstance(i, BatchNormStatsCallbak):
                i.fix_ownership(self)


class BiasRegularizer:
    """a contextmanager to be applied on a network that can encourage small
    biases (a.k.a. cardinality bound decay)"""

    _loss = 0
    _loss_max = 0
    _num_items = 0
    _device = None
    _bn_prev = None

    consider_sparsity = False

    def __init__(self, coeff: float, thresh: float, net: SeqBinModelHelper):
        self.coeff = coeff
        self.thresh = thresh
        self.net = net
        self._bn_prev = {}

        # find previous layers of all BNs
        prev = None
        for i in net.features:
            if isinstance(i, BatchNormStatsCallbak):
                assert isinstance(prev, (BinConv2d, BinLinear))
                self._bn_prev[i] = prev
            prev = i

    def on_bn_internals(self, bn, scale, bias):
        coeff = bn.bias_regularizer_coeff
        if coeff == 0:
            return

        cur = torch.relu(-bias / scale - self.thresh)
        if self.consider_sparsity:
            with torch.no_grad():
                weight = self._bn_prev[bn].weight_bin
                dim = bn.output_dim
                assert dim == weight.size(0)
                weight = (weight.view(dim, -1).abs() > 1e-4).to(weight.dtype)
                weight = weight.sum(dim=1).detach_()

            assert cur.size() == weight.size(), (cur.size(), weight.size())
            cur = cur * weight

        this_loss = cur.sum()
        self._num_items += cur.numel()
        if coeff != 1:
            this_loss *= coeff
        self._loss += this_loss
        with torch.no_grad():
            self._loss_max = torch.max(self._loss_max, cur.max()).detach_()


    def __enter__(self):
        if self._device is None:
            self._device = next(iter(self.net.parameters())).device
        self._loss = torch.tensor(0, dtype=torch.float32, device=self._device)
        self._loss_max = torch.tensor(0, dtype=torch.float32,
                                      device=self._device)
        self._num_items = 0
        if self.coeff:
            self.net.on_bn_internals = self.on_bn_internals

    def __exit__(self, exc_type, exc_value, traceback):
        if self.coeff:
            del self.net.on_bn_internals
            self._loss *= self.coeff

    @property
    def loss(self):
        return self._loss

    @property
    def loss_avg(self):
        assert self._num_items
        return self._loss / self._num_items

    @property
    def loss_max(self):
        return self._loss_max


class InputQuantizer(nn.Module):
    """quantize input in the range ``[0, 1]`` to be a multiple of given ``step``
    """

    class RoundFn(Function):
        @staticmethod
        def forward(ctx, inp):
            return torch.round(inp)

        @staticmethod
        def backward(ctx, g_out):
            return g_out


    def __init__(self, step: float):
        super().__init__()
        self.step = step

    def forward(self, x):
        if type(x) is AbstractTensor:
            return AbstractTensor(self.forward(x.vmin), self.forward(x.vmax),
                                  x.loss)

        if type(x) is MultiSampleTensor:
            return x.apply_batch(self.forward)

        xint = self.RoundFn.apply(x / self.step)
        return xint * self.step

    def __repr__(self):
        return f'{type(self).__name__}({self.step})'


class Mnist0(SeqBinModelHelper, nn.Module, ModelHelper):
    """see https://github.com/locuslab/convex_adversarial/blob/b3f532865cc02aeb6c020b449c2b1f0c13c7f236/examples/problems.py#L92"""

    CLASS2NAME = tuple(map(str, range(10)))

    def __init__(self, quant_step: float):
        super().__init__()
        self._setup_network(float(quant_step))

    def _setup_network(self, quant_step):
        self.make_small_network(self, quant_step, 1, 7)

    @classmethod
    def make_small_network(
            cls, self, quant_step, input_chl, pre_fc_spatial):

        conv = BinConv2dPos
        lin = BinLinearPos
        act = Binarize01Act
        wb = g_weight_binarizer

        chl = tuple(map(scale_channels, [16, 32, 100]))

        self.features = nn.Sequential(
            InputQuantizer(quant_step),
            # use bin conv on float inputs
            BinConv2d(wb, input_chl, chl[0], 4, stride=2, padding=1,
                      rounding=False),
            setattr_inplace(BatchNormStatsCallbak(self, chl[0]),
                            'bias_regularizer_coeff', 0),
            act(),

            conv(wb, chl[0], chl[1], 4, stride=2, padding=1),
            BatchNormStatsCallbak(self, chl[1]),
            act(),

            Flatten(),

            lin(wb, chl[1] * pre_fc_spatial**2, chl[2]),
            BatchNormStatsCallbak(self, chl[2]),
            act(),

            lin(wb, chl[2], 10),
            setattr_inplace(
                BatchNormStatsCallbak(
                    self, 10,
                    use_scalar_scale=g_use_scalar_scale_last_layer),
                'bias_regularizer_coeff', 0)
        )
        if g_remove_last_bn:
            self.features = self.features[:-1]

    @classmethod
    def make_dataset_loader(cls, args, train: bool):
        dataset = torchvision.datasets.MNIST(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batchsize, shuffle=train,
            num_workers=args.workers if train else 0)
        return loader


class MnistMLP(Mnist0):
    def _setup_network(self, quant_step):
        lin = BinLinearPos
        act = Binarize01Act
        wb = g_weight_binarizer

        layers = [
            InputQuantizer(quant_step),
            Flatten(),
        ]

        prev_chl = 28*28

        for chl in [500, 300, 200, 100, 10]:
            if len(layers) == 2:
                cur_lin = functools.partial(BinLinear, rounding=False)
            else:
                cur_lin = lin
            layers.extend([
                cur_lin(wb, prev_chl, chl),
                BatchNormStatsCallbak(self, chl),
                act(),
            ])
            prev_chl = chl

        del layers[-2:]
        # disable cardinality bound decay of input layer
        setattr_inplace(layers[3], 'bias_regularizer_coeff', 0)

        layers.append(
            setattr_inplace(
                BatchNormStatsCallbak(
                    self, 10,
                    use_scalar_scale=g_use_scalar_scale_last_layer),
                'bias_regularizer_coeff', 0
            ))
        if g_remove_last_bn:
            layers.pop()
        self.features = nn.Sequential(*layers)


class Mnist1(Mnist0):
    def _setup_network(self, quant_step):
        self.make_large_network(self, quant_step, 1, 7)

    @classmethod
    def make_large_network(
            cls, self, quant_step, input_chl, pre_fc_spatial,
            channel_sizes=[32, 64, 512, 512], kernel_sizes=[3, 4]):
        layers = [
            InputQuantizer(quant_step),
        ]
        prev_chl = input_chl

        conv = BinConv2dPos
        lin = BinLinearPos
        act = Binarize01Act
        wb = g_weight_binarizer

        # conv layers
        for out_chl in channel_sizes[:2]:
            for ksize in kernel_sizes:
                if len(layers) == 1:
                    # use bin conv on float inputs
                    cur_conv = functools.partial(BinConv2d, rounding=False)
                else:
                    cur_conv = conv
                layers.extend([
                    cur_conv(wb, prev_chl, out_chl, ksize,
                             stride=ksize-2, padding=1),
                    BatchNormStatsCallbak(self, out_chl),
                    act(),
                ])
                prev_chl = out_chl

        setattr_inplace(layers[2], 'bias_regularizer_coeff', 0)

        # fc layers
        layers.append(Flatten())
        prev_chl *= pre_fc_spatial**2
        for out_chl in channel_sizes[2:] + [10]:
            layers.extend([
                lin(wb, prev_chl, out_chl),
                BatchNormStatsCallbak(self, out_chl),
                act(),
            ])
            prev_chl = out_chl

        del layers[-2:]
        layers.append(
            setattr_inplace(
                BatchNormStatsCallbak(self, 10, use_scalar_scale=True),
                'bias_regularizer_coeff', 0
            ))
        self.features = nn.Sequential(*layers)


class Cifar10_0(SeqBinModelHelper, nn.Module, ModelHelper):
    CLASS2NAME = (
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck',
    )

    def __init__(self, quant_step: float):
        super().__init__()
        self._setup_network(float(quant_step))

    def _setup_network(self, quant_step):
        Mnist0.make_small_network(self, quant_step, 3, 8)

    @classmethod
    def make_dataset_loader(self, args, train: bool):
        dataset = torchvision.datasets.CIFAR10(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batchsize, shuffle=train,
            num_workers=args.workers if train else 0)
        return loader


class Cifar10_1(Cifar10_0):
    def _setup_network(self, quant_step):
        Mnist1.make_large_network(self, quant_step, 3, 8)

from .net_real import MnistReal0, MnistReal1, Cifar10Real0, Cifar10Real1
