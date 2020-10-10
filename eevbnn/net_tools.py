from .net_bin import BiasRegularizer, BatchNormStatsCallbak
from .utils import (get_nr_correct, default_dataset_root, ModelHelper,
                    ensure_training_state)

import torch
from tqdm import tqdm

import argparse

def eval_cbd_stats(args, net, device):
    """evluate CBD loss stats

    :return: avg, max loss"""
    assert not net.training
    testloader = net.make_dataset_loader(args, False)
    cbd = BiasRegularizer(1, 0, net)

    tot_loss = 0
    tot_num = 0
    max_loss = 0

    nr_correct = 0
    with ensure_training_state(net, False):
        for inputs, labels in testloader:
            with cbd:
                out = net(inputs.to(device))
            nr = inputs.size(0)
            tot_loss += float(cbd.loss_avg.item() * nr)
            tot_num += nr
            max_loss = max(max_loss, float(cbd.loss_max.item()))
            nr_correct += get_nr_correct(out, labels.to(device))

    avg = float(tot_loss / tot_num)
    print(f'CBD: {avg=:.2f} max={max_loss:.2f} '
          f'acc={nr_correct/tot_num*100:.2f}%')
    return avg, max_loss

class OnlineMean:
    """compute the weighted mean in an online manner"""
    _acc = None
    _wsum = None

    def add(self, x: torch.Tensor, weight: float = 1):
        if self._acc is None:
            self._acc = x.clone()
            self._wsum = weight
            return

        ws_next = self._wsum + weight
        k = weight / ws_next
        self._acc = self._acc.mul_(1 - k).add_(x * k)
        self._wsum = ws_next

    def get(self):
        return self._acc

def callib_bn(args, net, device):
    def update_bn_stat(ftr, bn, bn_idx):
        ref_batch = None
        reshape_brd = None

        def process(name, remap, scalar):
            nonlocal ref_batch
            nonlocal reshape_brd

            ret = OnlineMean()
            tot = 0

            for inputs, _ in tqdm(net.make_dataset_loader(args, True),
                                  f'bn-{bn_idx}:{name}'):
                out = ftr(inputs.to(device))

                if ref_batch is None:
                    ref_batch = inputs.size(0)
                    if out.ndim == 2:
                        reshape_brd = [1, out.size(1)]
                    else:
                        assert out.ndim == 4
                        reshape_brd = [1, out.size(1), 1, 1]

                w = inputs.size(0) / ref_batch

                if scalar:
                    cur = remap(out).mean()
                elif out.ndim == 2:
                    cur = remap(out).mean([0])
                else:
                    cur = remap(out).mean([0, 2, 3])
                ret.add(cur, w)
                tot += out.numel() // cur.numel()
            return ret, tot

        mean, _ = process('mean', lambda x: x, False)
        mean_brd = mean.get().view(reshape_brd)
        var, nr = process('var',
                          lambda x: (x - mean_brd)**2, bn.use_scalar_scale)
        bn.running_mean.copy_(mean.get())
        bn.running_var.copy_(var.get() * (nr / (nr-1)))

    with torch.no_grad():
        with ensure_training_state(net, False):
            for idx, layer in enumerate(net.features):
                if isinstance(layer, BatchNormStatsCallbak):
                    update_bn_stat(net.features[:idx], layer, idx)


def callib_bn_main(args, net, device):
    eval_cbd_stats(args, net, device)
    callib_bn(args, net, device)
    eval_cbd_stats(args, net, device)
    if args.output:
        net.save_to_file(args.output)

FN_MAP = {
    'eval_cbd': eval_cbd_stats,
    'calc_bn': callib_bn_main,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('net')
    parser.add_argument('cmd', help='command to run', choices=FN_MAP.keys())
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument(
        '--data', default=default_dataset_root(),
        help='dir for training data')
    parser.add_argument('-o', '--output',
                        help='write the network with callibrated BN')
    parser.add_argument('--workers', type=int, default=2,
                        help='number of CPU workers for data loading')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ModelHelper.create_with_load(args.net).to(device)
    FN_MAP[args.cmd](args, net, device)

if __name__ == '__main__':
    main()
