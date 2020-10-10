#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import shlex
import os
from pathlib import Path

DEFAULT_TIMEOUT = 120

def setup_affinity(self_id, tot):
    assert 0 <= self_id < tot
    nr_cpu = os.cpu_count()
    begin = (self_id * nr_cpu) // tot
    end = ((self_id + 1) * nr_cpu) // tot
    os.sched_setaffinity(0, range(begin, end))

def main():
    parser = argparse.ArgumentParser(
        description='run bin eval with given args'
    )
    parser.add_argument('gpu_list', help='available GPU IDs list')
    parser.add_argument('output_dir', help='training output directory')
    parser.add_argument('nr_jobs', type=int, help='total number of jobs')
    parser.add_argument('slot_id', type=int,
                        help='job slot ID to decide the GPU')
    parser.add_argument('args',
                        help='one line from the task file for args, in the '
                        'format dir_name eps [out_file_name [other args ...]]')
    args = parser.parse_args()

    gpus = args.gpu_list.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus[args.slot_id % len(gpus)]

    replace_dict = {
        '$dir': args.output_dir,
        '$2/255': '0.00784313725490196',
        '$20/255': '0.0784313725490196',
        '$5/255': '0.0196078431372549',
        '$8/255': '0.03137254901960784',
        '$SUBSET': '--timeout 3600 --random-sample 40',
    }

    extra_args = args.args
    for k, v in replace_dict.items():
        extra_args = extra_args.replace(k, v)
    dir_name, eps, *other = shlex.split(extra_args)
    if other and not other[0].startswith('-'):
        out_file_name, *other = other
    else:
        out_file_name = 'minisatcs-verify'

    if '--timeout' not in other:
        tle = ['--timeout', str(DEFAULT_TIMEOUT)]
    else:
        tle = []

    work_dir = Path(args.output_dir) / dir_name
    if not (work_dir / 'finish_mark').exists():
        print(f'skip unfinished task {work_dir}')
        return

    full_out_name = f'eval-{out_file_name}-{eps}'

    if (work_dir / f'{full_out_name}.json.finished').exists():
        print(f'skip finished task {full_out_name} in {work_dir}')
        return

    sub_args = (
        ['python', '-m', 'eevbnn', 'eval_bin', str(work_dir / 'last.pth'),
         '-e', eps, '--check-cvt',
         '--write-result', str(work_dir / f'{full_out_name}.json'),
         '--log-file', str(work_dir / f'{full_out_name}.txt'),
         ] + tle + other
    )
    if '--skip' in other:
        print('would not continue since --skip is provided')
    else:
        other.append('--continue')

    setup_affinity(args.slot_id - 1, args.nr_jobs)
    os.execve(sys.executable, sub_args, os.environ)

if __name__ == '__main__':
    main()
