#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=$1 python tools/test.py $2 $3 --out 'work_dirs/out.pkl' --show-dir 'work_dirs/' --eval bbox