#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import eevbnn.net_bin as net_bin
from eevbnn.utils import (default_dataset_root, npy_chw_f32_to_cv2_uint8,
                          torch_as_npy, ensure_dir)
import numpy as np
import torch
import matplotlib.pyplot as plt

import argparse
import json
import cv2

def plot_img(class2name, out_dir, result):
    result = list(result.values())
    fig_cols = 6
    fig_rows = min(len(result) // fig_cols, 6)
    if (nr := fig_cols * fig_rows) < len(result):
        np.random.RandomState(42).shuffle(result)
        result = result[:nr]


    fig, axes = plt.subplots(
        fig_rows, fig_cols,
        gridspec_kw={'wspace': 0.05, 'hspace': 0.4},
        squeeze=True,
        figsize=(fig_cols*1.6 + (fig_cols-1)*0.05,
                 fig_rows*1.6 + (fig_rows-1)*0.4))

    for r in range(fig_rows):
        for c in range(fig_cols):
            ax = axes[r, c]
            cur = result[r * fig_cols + c]
            img = cur['img']
            if img.shape[2] == 1:
                img = img[:, :, 0]
            ax.imshow(img, interpolation=None)
            ax.axis('off')
            ax.set_title(f' pred={class2name[cur["prediction"]]}\n'
                         f'label={class2name[cur["label"]]}')
    fig.savefig(str(out_dir / 'img.pdf'))

def main():
    parser = argparse.ArgumentParser(
        description='save images that have different verification results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('network', help='network file')
    parser.add_argument('vrf1', help='first verification result')
    parser.add_argument('vrf2', help='second verification result')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument(
        '--data', default=default_dataset_root(),
        help='dir for training data')
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net_bin.ModelHelper.create_with_load(args.network).to(device)
    with open(args.vrf1) as fin:
        vrf1: dict = json.load(fin)
    with open(args.vrf2) as fin:
        vrf2: dict = json.load(fin)

    result = {}
    cnt_num = -1
    out_dir = Path(args.out_dir)
    cv_images = {}
    for inputs, labels in net.make_dataset_loader(args, False):
        for img, lbl in zip(inputs, labels):
            cnt_num += 1
            cnt = str(cnt_num)
            v1 = vrf1.get(cnt)
            v2 = vrf2.get(cnt)
            if v1 is None or v2 is None:
                continue
            if v1['result'] != v2['result']:
                cv_img = npy_chw_f32_to_cv2_uint8(torch_as_npy(img))
                cv2.imwrite(str(out_dir / f'{cnt}.png'), cv_img)
                cv_images[cnt] = cv_img
                pred = int(
                    torch.argmax(
                        net(img[np.newaxis].to(device)), dim=1).item()
                )
                result[cnt] = {
                    'prediction': pred,
                    'label': int(lbl.item()),
                    'vrf1': v1['result'],
                    'vrf2': v2['result'],
                }

    with (out_dir / 'result.json').open('w') as fout:
        json.dump(result, fout, indent=2)

    print(f'num images: {len(result)}')
    for k, v in result.items():
        v['img'] = cv_images[k]

    plot_img(net.CLASS2NAME, out_dir, result)


if __name__ == '__main__':
    main()
