#!/bin/bash -e

out_dir="$1"

if [ -z "$out_dir" ]; then
    echo "usage: $0 <training output dir>"
    exit 1
fi

[ -z "$GPU_LIST" ] && GPU_LIST=0,1

exec parallel -j6 --lb --termseq INT,500,INT,500,INT,2000,KILL,25 --eta \
    "./attack_runner.py $GPU_LIST $out_dir {%} {}" \
    :::: attack_all_task.txt
