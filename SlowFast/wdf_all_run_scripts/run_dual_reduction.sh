#!/bin/bash

# 2020 07 21
# slowfast dual attenrion
# 24 * 224, 32 frame
# ALPHA: 4
# BETA_INV: 8
# add reduction to save memory in each fusion layer.

ROOT_PATH=/data1/SlowFast_vis_0709/SlowFast
CONFIG_YAML=configs/Kinetics/SLOWFAST_DUAL_8x8_R50_stepwise_multigrid.yaml

python $ROOT_PATH/tools/run_net.py --cfg $ROOT_PATH/$CONFIG_YAML NUM_GPUS 8  TRAIN.BATCH_SIZE 16  TEST.BATCH_SIZE 16  DATA_LOADER.NUM_WORKERS 16
