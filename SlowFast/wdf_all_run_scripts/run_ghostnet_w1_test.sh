#!/bin/bash

# 2020 08 09
# slowfast ghost net
# 112 * 112, 16 frame
# width = 1.0

ROOT_PATH=/data1/SlowFast_vis_0709/SlowFast
CONFIG_YAML=configs/Kinetics/SLOWFAST_GHOSTNET_8x8_R50_stepwise_multigrid.yaml

python $ROOT_PATH/tools/run_net.py --cfg $ROOT_PATH/$CONFIG_YAML NUM_GPUS 4  TRAIN.BATCH_SIZE 16 TEST.BATCH_SIZE 16  DATA_LOADER.NUM_WORKERS 16 SLOWFAST.WIDTH_MULTI 1.0 SOLVER.MAX_EPOCH 100  TRAIN.ENABLE False MULTIGRID.SHORT_CYCLE False MULTIGRID.LONG_CYCLE False TEST.CHECKPOINT_FILE_PATH /data1/ADAS/KINETICS_SlowFastGhostNet_W1/checkpoints/checkpoint_epoch_00141.pyth 

#python tools/run_net.py  --cfg configs/Kinetics/SLOWFAST_GHOSTNET_8x8_R50_stepwise_multigrid.yaml SLOWFAST.WIDTH_MULTI 1.0 OUTPUT_DIR /data1/ADAS/KINETICS_SlowFastGhostNet_W1  SOLVER.MAX_EPOCH 100  TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH /data1/ADAS/KINETICS_SlowFastGhostNet_W1/checkpoints/checkpoint_epoch_00141.pyth TEST.BATCH_SIZE 64 MULTIGRID.SHORT_CYCLE False MULTIGRID.LONG_CYCLE False 
