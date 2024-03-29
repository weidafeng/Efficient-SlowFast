#!/bin/bash

# 2020 08 13
# slowfast shuffle net
# 112 * 112, 16 frame
# width = 2.0
# group = 3

ROOT_PATH=/data1/SlowFast_vis_0709/SlowFast
CONFIG_YAML=configs/Kinetics/SLOWFAST_MOBILENETV2_8x8_R50_stepwise_multigrid.yaml 

python $ROOT_PATH/tools/run_net.py --cfg $ROOT_PATH/$CONFIG_YAML NUM_GPUS 2  TRAIN.BATCH_SIZE 128 TEST.BATCH_SIZE 128  DATA_LOADER.NUM_WORKERS 16 OUTPUT_DIR /data1/ADAS/KINETICS_SlowFastMobileNetV2_W1 SLOWFAST.WIDTH_MULTI 1.0  SOLVER.MAX_EPOCH 100 MULTIGRID.SHORT_CYCLE False MULTIGRID.LONG_CYCLE False TRAIN.CHECKPOINT_PERIOD 3 
