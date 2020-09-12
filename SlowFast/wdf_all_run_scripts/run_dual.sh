#!/bin/bash

# 2020 07 21
# slowfast dual attenrion
# 24 * 224, 32 frame
# ALPHA: 4
# BETA_INV: 8

ROOT_PATH=/data1/SlowFast_vis_0709/SlowFast
CONFIG_YAML=configs/Kinetics/SLOWFAST_DUAL_8x8_R50_stepwise_multigrid.yaml

python $ROOT_PATH/tools/run_net.py --cfg $ROOT_PATH/$CONFIG_YAML NUM_GPUS 4  TRAIN.BATCH_SIZE 4  TEST.BATCH_SIZE 4  DATA_LOADER.NUM_WORKERS 8 SOLVER.MAX_EPOCH 100 OUTPUT_DIR /data1/ADAS/KINETICS_SlowFastDUAL 
