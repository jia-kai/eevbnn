from .utils import make_grid_to_cv2uint8, ModelHelper, default_dataset_root
from .attack_impl import fgsm_batch, pgd_batch, loss_fn_map
from . import net_bin

import torch
import torchvision
import numpy as np
import cv2
from tqdm import tqdm
from tabulate import tabulate

from pathlib import Path
import pickle
import sys
import json
import argparse

def show_adv(img, img_adv, eps):
    cv2.imshow('input', make_grid_to_cv2uint8(img, normalize=False))
    cv2.imshow(f'adv{eps}', make_grid_to_cv2uint8(img_adv, normalize=False))
    key = chr(cv2.waitKey(-1) & 0xff)
    cv2.destroyAllWindows()
    if key == 'q':
        sys.exit(0)

def eval_tanh_scale(args, net, device, loss_fn, eps):
    all_training_inputs = []
    all_training_labels = []
    for i, j in net.make_dataset_loader(args, True):
        all_training_inputs.append(i)
        all_training_labels.append(j)
    all_training_inputs = torch.cat(all_training_inputs)
    all_training_labels = torch.cat(all_training_labels)
    idx = torch.randperm(all_training_inputs.shape[0])[:args.batchsize]

    inputs = all_training_inputs[idx].to(device)
    labels = all_training_labels[idx].to(device)

    low = 0.5
    high = 3

    def get_acc(val):
        net_bin.g_bingrad_soft_tanh_scale = val
        img, cur_correct = pgd_batch(
            net, inputs, loss_fn(labels), eps,
            nr_step=args.pgd_steps)
        return cur_correct

    acc2scale = []
    for s in np.arange(0.5, 3, 0.1):
        print(f'{s:.2f}', acc := get_acc(s))
        acc2scale.append((acc, s))
    acc2scale.sort()
    print('===============')
    for acc, scale in acc2scale:
        print(f'{scale:.2f}', acc)

def main():
    parser = argparse.ArgumentParser(
        description='run FGSM/PGD attack on a model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('net', help='name of network')
    parser.add_argument('--batchsize', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4,
                        help='number of CPU workers for data augmentation')
    parser.add_argument(
        '--data', default=default_dataset_root(), help='dir for training data')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='show pics of adv examples on a batch of many wrong predictions')
    parser.add_argument(
        '--verbose-thresh', type=float, default=0.3,
        help='batch to show if precision is below this threshold')
    parser.add_argument('--pgd', action='store_true',
                        help='enable PGD attack')
    parser.add_argument('--pgd-steps', type=int, default=10,
                        help='number of PGD attack steps')
    parser.add_argument('-e', '--eps', type=float, default=[], action='append',
                        required=True,
                        help='attack eps; multiple values can be given')
    parser.add_argument('-l', '--loss', default='xent',
                        choices=loss_fn_map.keys(), help='loss function to use')
    parser.add_argument('--debug-input',
                        help='attack on a single input data for debug; must be '
                        'a pickled dict containing keys "inp", "label" and '
                        '"inp_adv"')
    parser.add_argument('--write-result',
                        help='write attack result to a json file')
    parser.add_argument('--eval-tanh-scale', action='store_true',
                        help='eval tanh scale using provided network')
    parser.add_argument('--use-hardtanh-grad', action='store_true',
                        help='use hardtanh for gradient computing')
    args = parser.parse_args()

    if args.use_hardtanh_grad:
        net_bin.g_bingrad_soft_tanh_scale = None


    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ModelHelper.create_with_load(args.net).to(device)
    loss_fn = loss_fn_map[args.loss]

    if args.verbose:
        print(f'network: {net}')

    eps = args.eps
    if args.eval_tanh_scale:
        assert len(eps) == 1, 'only one eps should be provided'
        return eval_tanh_scale(args, net, device, loss_fn, eps[0])

    inp_adv = None
    if args.debug_input:
        with open(args.debug_input, 'rb') as fin:
            data_pack = pickle.load(fin)

        inp = np.ascontiguousarray(data_pack['inp'], dtype=np.float32)
        label = np.ascontiguousarray(data_pack['label'], dtype=np.int64)
        inp_adv = np.ascontiguousarray(data_pack['inp_adv'], dtype=np.float32)
        if inp.ndim == 3:
            inp = inp[np.newaxis]
            inp_adv = inp_adv[np.newaxis]
        assert (inp.ndim == 4 and label.ndim == 1 and inp.shape == inp_adv.shape
                and inp.shape[0] == label.shape[0])
        inp, label, inp_adv = map(torch.from_numpy, (inp, label, inp_adv))
        inp_adv = inp_adv.to(device)
        testloader = [(inp, label)]
    else:
        testloader = net.make_dataset_loader(args, train=False)


    result_dict = {}
    def run_fgsm():
        shown = [not args.verbose] * len(eps)
        print('======= fgsm =======')
        out_tbl = [['eps', 'correct']]
        nr_correct0 = 0
        nr_correct_eps = np.zeros(len(eps))
        tot = 0
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            cur_correct, attack_results = fgsm_batch(
                net, inputs, loss_fn(labels), eps)
            nr_correct0 += cur_correct
            nr_correct_eps += [i[1] for i in attack_results]
            tot += inputs.size(0)
            for i, j in enumerate(shown):
                if (not j and
                        attack_results[i][1] < inputs.size(0) *
                        args.verbose_thresh):
                    shown[i] = True
                    show_adv(inputs, attack_results[i][0], eps[i])

        out_tbl.append([0, nr_correct0 / tot])
        out_tbl.extend(([i, j / tot] for i, j in zip(eps, nr_correct_eps)))
        result_dict['fgsm'] = dict(out_tbl[2:])

        print(tabulate(out_tbl, headers='firstrow'))
        return tot

    def run_pgd(tot):
        print('======= pgd =======')
        out_tbl = [['eps', 'correct']]
        result_dict['pgd'] = {}
        result_dict['pgd_steps'] = args.pgd_steps
        with tqdm(total=tot * len(eps)) as pbar:
            for cur_eps in eps:
                shown = not args.verbose
                tot = 0
                nr_correct = 0
                for inputs, labels in testloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    img, cur_correct = pgd_batch(
                        net, inputs, loss_fn(labels), cur_eps,
                        nr_step=args.pgd_steps, dbg_known_adv=inp_adv)
                    nr_correct += cur_correct
                    tot += inputs.size(0)
                    pbar.update(inputs.size(0))

                    if (not shown and
                            cur_correct < inputs.size(0) * args.verbose_thresh):
                        shown = True
                        show_adv(inputs, img, cur_eps)

                out_tbl.append([cur_eps, nr_correct / tot])
                result_dict['pgd'][cur_eps] = nr_correct / tot

        print(tabulate(out_tbl, headers='firstrow'))

    tot = run_fgsm()
    if args.pgd:
        run_pgd(tot)

    if args.write_result:
        with open(args.write_result, 'w') as fout:
            json.dump(result_dict, fout, indent=2)

if __name__ == '__main__':
    main()
