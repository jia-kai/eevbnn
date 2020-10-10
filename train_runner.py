#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import shlex
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='run bin training with given args'
    )
    parser.add_argument('gpu_list', help='available GPU IDs list')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('slot_id', type=int,
                        help='job slot ID to decide the GPU')
    parser.add_argument('args',
                        help='one line from the task file for args, in the '
                        'format net_name out_suffix additional_args...')
    args = parser.parse_args()
    gpus = args.gpu_list.split(',')
    real_args = shlex.split(args.args.replace(
        '$dir', args.output_dir
    ))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpus[args.slot_id % len(gpus)]
    out_dir = Path(args.output_dir) / real_args[1]
    if (out_dir / 'finish_mark').exists():
        print(f'skip finished task {out_dir}')
        return
    sub_args = (
        ['python', '-m', 'eevbnn', 'train_bin', real_args[0], str(out_dir)] +
        real_args[2:]
    )

    os.execve(sys.executable, sub_args, os.environ)

if __name__ == '__main__':
    main()
