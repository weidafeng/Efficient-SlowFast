#!/bin/bash

# 2020 07 21
# slowfast mobilenet v2
# 112 * 122, 16 frame
# ALPHA: 4
# BETA_INV: 8
# width: 0.5

ROOT_PATH=/data1/SlowFast_vis_0709/SlowFast
CONFIG_YAML=configs/Kinetics/SLOWFAST_MOBILENETV2_8x8_R50_stepwise_multigrid.yaml

python $ROOT_PATH/tools/run_net.py --cfg $ROOT_PATH/$CONFIG_YAML NUM_GPUS 8  TRAIN.BATCH_SIZE 512  TEST.BATCH_SIZE 256  DATA_LOADER.NUM_WORKERS 16
