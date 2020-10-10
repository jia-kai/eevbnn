#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json

def main():
    parser = argparse.ArgumentParser(
        description='merge multiple eval result files (manual sharding by '
        'parallel jobs)'
    )
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('inputs', nargs='+')
    args = parser.parse_args()

    merged = {}
    for i in args.inputs:
        with open(i) as fin:
            cur = json.load(fin)
        merged.update(cur)

    print(f'merged size: {len(merged)}')
    with open(args.output, 'w') as fout:
        json.dump(merged, fout)

if __name__ == '__main__':
    main()
