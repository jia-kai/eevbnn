#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description='analyze statistics from evaluation result'
    )
    parser.add_argument(
        'results', nargs='+',
        help='json log results; multiple files can be given; it can also be a '
        'directory where all json files should be checked')
    parser.add_argument('--check-consist', action='store_true',
                        help='check if all given results are consistent')

    args = parser.parse_args()

    if len(args.results) == 1 and (p := Path(args.results[0])).is_dir():
        args.results = [i for i in p.iterdir() if i.name.endswith('.json')]

    merged_result = {}
    for fname in args.results:
        with open(fname) as fin:
            data: dict = json.load(fin)

        if args.check_consist:
            for k, v in data.items():
                v = v['result']
                old_v = merged_result.setdefault(k, v)
                if old_v != v:
                    assert old_v == 'TLE' or v == 'TLE', (
                        f'inconsistency: {old_v} vs {v} from {fname}'
                    )
                    if old_v == 'TLE':
                        merged_result[k] = v

        result_solved = np.array([i['result'] != 'TLE' for i in data.values()],
                                 dtype=np.int32)
        result_rob = np.array([i['result'] == 'UNSAT' for i in data.values()],
                              dtype=np.int32)
        result_time = np.array([i['solve_time'] for i in data.values()],
                               dtype=np.float32)

        def print_time(title, time):
            if not len(time):
                return
            time = np.ascontiguousarray(time, dtype=np.float32)
            print(f'  {title}: mean={time.mean():.7f} max={time.max():.3f} '
                  f'med={np.median(time):.3f}')

        print(f'{fname} ========= len={len(data)}')
        print_time('time', result_time)
        print_time('build time', [i['build_time'] for i in data.values()])
        print(f'  robust: {result_rob.mean()*100:.2f}% '
              f'({result_rob.sum()})')
        print(f'  solved: {result_solved.mean()*100:.2f}% '
              f'({result_solved.sum()})')
        print_time('time on robust',
                   [result_time[i] for i in range(result_time.size)
                    if result_rob[i]])

if __name__ == '__main__':
    main()
