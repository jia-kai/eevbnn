#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import shlex
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='run PGD attack with given args'
    )
    parser.add_argument('gpu_list', help='available GPU IDs list')
    parser.add_argument('output_dir', help='training output directory')
    parser.add_argument('slot_id', type=int,
                        help='job slot ID to decide the GPU')
    parser.add_argument('args',
                        help='one line from the task file for args, in the '
                        'format dir_name eps,eps,...,eps [hardtanh]')
    args = parser.parse_args()

    gpus = args.gpu_list.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus[args.slot_id % len(gpus)]

    dir_name, eps, *hardtanh = shlex.split(args.args)
    extra_args = []

    if hardtanh:
        assert hardtanh == ['hardtanh']
        outname = 'attack-hardtanh.json'
        extra_args.append('--use-hardtanh-grad')
    else:
        outname = 'attack.json'

    eps = eps.split(',')
    for i in eps:
        extra_args.extend(['-e', i])

    work_dir = Path(args.output_dir) / dir_name
    if not (work_dir / 'finish_mark').exists():
        print(f'skip unfinished task {work_dir}')
        return

    if (work_dir / outname).exists():
        print(f'skip finished task {outname} in {work_dir}')
        return

    sub_args = (
        ['python', '-m', 'eevbnn', 'attack', str(work_dir / 'last.pth'),
         '--write-result', str(work_dir / outname),
         '--pgd-steps=100', '--pgd',
         ] + extra_args
    )

    os.execve(sys.executable, sub_args, os.environ)

if __name__ == '__main__':
    main()

