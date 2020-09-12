#!/bin/bash

# 2020 07 21
# slowfast shuffle net v2
# 112 * 112, 16 frame
# width = 0.25

ROOT_PATH=/data1/SlowFast_vis_0709/SlowFast
CONFIG_YAML=configs/Kinetics/SLOWFAST_SHUFFLENETV2_8x8_R50_stepwise_multigrid.yaml

python $ROOT_PATH/tools/run_net.py --cfg $ROOT_PATH/$CONFIG_YAML NUM_GPUS 4  TRAIN.BATCH_SIZE 256  DATA_LOADER.NUM_WORKERS 16
