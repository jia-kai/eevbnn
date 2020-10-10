#!/bin/bash -e

set -u -e

cd $(dirname $0)
git submodule update --init roundingsat_repo
cd roundingsat_repo

if [ ! -f soplex-5.0.1.tgz ]; then
    echo "Please down soplex-5.0.1 from https://soplex.zib.de/download.php?fname=soplex-5.0.1.tgz and place it at $(pwd)"
    exit -1
fi

[ -f build/roundingsat ] && exit 0

cd build
cmake  -DCMAKE_BUILD_TYPE=Release -Dsoplex=ON ..
make -j$(nproc)
