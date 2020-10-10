from .utils import ModelHelper, Flatten

import torch.nn as nn

class SeqRealModelHelper:
    def forward(self, x):
        return self.features(x)

    def get_sparsity_stat(self):
        pass


class MnistReal0(SeqRealModelHelper, nn.Module, ModelHelper):
    """real-valued mnist for reference"""
    def __init__(self, quant_step: float):
        super().__init__()
        self._setup_network(float(quant_step))

    def _setup_network(self, quant_step):
        self.make_small_network(self, quant_step, 1, 7)

    @classmethod
    def make_small_network(cls, self, quant_step, input_chl, pre_fc_spatial):
        self.features = nn.Sequential(
            InputQuantizer(quant_step),
            nn.Conv2d(input_chl, 16, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),

            Flatten(),

            nn.Linear(32 * pre_fc_spatial**2, 100),
            nn.ReLU(),

            nn.Linear(100, 10),
        )

    def forward(self, x):
        return self.features(x)

    def make_dataset_loader(self, args, train: bool):
        return Mnist0.make_dataset_loader(args, train)


class MnistReal1(MnistReal0):
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

        # conv layers
        for out_chl in channel_sizes[:2]:
            for ksize in kernel_sizes:
                layers.extend([
                    nn.Conv2d(prev_chl, out_chl, ksize,
                             stride=ksize-2, padding=1),
                    nn.ReLU(),
                ])
                prev_chl = out_chl

        # fc layers
        layers.append(Flatten())
        prev_chl *= pre_fc_spatial**2
        for out_chl in channel_sizes[2:] + [10]:
            layers.extend([
                nn.Linear(prev_chl, out_chl),
                nn.ReLU(),
            ])
            prev_chl = out_chl

        layers.pop()    # remove last relu
        self.features = nn.Sequential(*layers)


class Cifar10Real0(SeqRealModelHelper, nn.Module, ModelHelper):
    def __init__(self, quant_step: float):
        super().__init__()
        self._setup_network(float(quant_step))

    def _setup_network(self, quant_step):
        MnistReal0.make_small_network(self, quant_step, 3, 8)

    @classmethod
    def make_dataset_loader(self, args, train: bool):
        return Cifar10_0.make_dataset_loader(args, train)


class Cifar10Real1(Cifar10Real0):
    def _setup_network(self, quant_step):
        MnistReal1.make_large_network(self, quant_step, 3, 8)

from .net_bin import InputQuantizer, Mnist0, Cifar10_0
