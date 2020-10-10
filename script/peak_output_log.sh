#!/bin/bash -e

for fname in "$@"; do
    [ -d $fname ] && fname="$fname/log.txt"
    echo -e "============ $fname"
    grep bloss $fname | tail -n 1
    grep -E "^(test|sparsity)" $fname | tail -n 4
    best=$(grep 'best model' $fname | tail -n 1)
    if [ -n "$best" ]; then
        acc=$(grep "test$(echo $best | grep -o '@[0-9]*')" $fname)
        echo "$best; $acc"
    fi
    echo
done
