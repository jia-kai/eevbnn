#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from eevbnn.utils import ModelHelper, default_dataset_root
import torch

import argparse
import time

def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(
        description='load a model and test its CPU inference time on a '
        'single image'
    )
    parser.add_argument('model')

    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8,
                        help='number of CPU workers for data augmentation')
    parser.add_argument(
        '--data', default=default_dataset_root(),
        help='dir for training data')

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ModelHelper.create_with_load(args.model).to(device).eval()
    testloader = model.make_dataset_loader(args, False)
    imgs, _ = next(iter(testloader))
    imgs = imgs.to(device)
    img = imgs[:1]
    traced = torch.jit.trace(model, (img, ))

    img = imgs[1:2]
    model(img)
    traced(img)

    t0 = time.time()
    model(img)
    print(f'model time: {time.time() - t0}')

    t0 = time.time()
    traced(img)
    print(f'traced time: {time.time() - t0}')

if __name__ == '__main__':
    main()
