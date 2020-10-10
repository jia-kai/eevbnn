#!/bin/bash -e

dir=$1

if [ -z "$dir" ]; then
    echo "usage: $0 <training dir>"
    exit -1
fi

cd $(dirname $0)/..

./script/eval_stat.py --check \
    $dir/mnist-mlp/eval-{minisatcs,m22{,-cardnet}}-0.0784313725490196.json

echo '=============='

./script/eval_stat.py --check \
    $dir/mnist-s-adv0.3/eval-{minisatcs-verify,m22{,-cardnet}}-0.3.json

echo '=============='

./script/eval_stat.py --check \
    $dir/mnist-l-adv0.3-cbd3/eval-{minisatcs-verify,m22{,-cardnet}}-0.3.json
