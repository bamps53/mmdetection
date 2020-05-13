#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_train.sh $1 1 --validate
CUDA_VISIBLE_DEVICES=0 PORT=29500 python tools/test_eval.py $1