#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shlex
import sys
from pprint import pformat
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='check whether trained models use the same command line as '
        'specified in task file'
    )
    parser.add_argument('spec')
    parser.add_argument('train_dir')
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    unused = set(i.name for i in train_dir.iterdir() if i.is_dir())
    exitcode = 0
    nr_fail = 0
    nr_finished = 0
    tot = 0
    with open(args.spec) as fin:
        for line in fin:
            line = line.replace('$dir', args.train_dir)
            tot += 1
            net, out_dir, *spec = shlex.split(line.strip())
            try:
                unused.remove(out_dir)
            except:
                pass
            spec.insert(0, net)
            log_path = train_dir / out_dir / 'log.txt'
            if not log_path.exists():
                print(f'{log_path} does not exist')
                exitcode |= 2
                nr_fail += 1
                continue

            if (train_dir / out_dir / 'finish_mark').exists():
                nr_finished += 1

            with log_path.open() as flog:
                got = flog.readline().strip()

            assert got.startswith('argv:'), f'bad {out_dir}'
            got = eval(got[5:])[1:]
            del got[1]  # output dir
            if got != spec:
                print(f'=== mismatch {out_dir}')
                print(f'actual: {pformat(got)}')
                print(f'spec:   {pformat(spec)}')
                exitcode |= 1
                nr_fail += 1

    print(f'{tot} spec checked: failed={nr_fail} finished={nr_finished}')
    if unused:
        print(f'unused output dirs: {unused}')
        exitcode |= 3
    sys.exit(exitcode)

if __name__ == '__main__':
    main()
