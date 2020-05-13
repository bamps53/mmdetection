#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=1 PORT=29501 ./tools/dist_train.sh $1 1 --validate
#CUDA_VISIBLE_DEVICES=1 PORT=29501 python tools/test_eval.py $1
