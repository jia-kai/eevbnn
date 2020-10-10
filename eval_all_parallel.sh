#!/bin/bash -e

out_dir="$1"
task_file="$2"

if [ -z "$out_dir" ]; then
    echo "usage: $0 <training output dir> [<task file>]"
    exit 1
fi

[ -z "$task_file" ] && task_file=eval_all_task.txt
[ -z "$GPU_LIST" ] && GPU_LIST=0,0,1
[ -z "$JOBS" ] && JOBS=12

exec parallel -j$JOBS --lb --termseq INT,500,INT,500,INT,2000,KILL,25 --eta \
    "./eval_runner.py $GPU_LIST $out_dir $JOBS {%} {}" \
    :::: $task_file

