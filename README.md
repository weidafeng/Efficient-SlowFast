# Efficient dual attention SlowFast networks for video action recognition

Dafeng Wei,  Ye Tian,  Liqing Wei,  Hong Zhong,   Siqian Chen,   **Shiliang Pu,    Hongtao Lu** (Corresponding authors)



## Abstract

```bash
Video data mainly differ in temporal dimension compared with static image data. Various video action recognition networks choose two-stream models to learn spatial and temporal information separately and fuse them to further improve performance. We proposed a cross-modality dual attention fusion module named CMDA to explicitly exchange spatial–temporal information between two pathways in two-stream SlowFast networks. Besides, considering the computational complexity of these heavy models and the low accuracy of existing lightweight models, we proposed several two-stream efficient SlowFast networks based on well- designed efficient 2D networks, such as GhostNet, ShuffleNetV2 and so on. Experiments demonstrate that our proposed fusion model CMDA improves the performance of SlowFast, and our efficient two-stream models achieve a consistent increase in accuracy with a little overhead in FLOPs. Our code and pre-trained models will be made available at https://github.com/weidafeng/Efficient-SlowFast
```



## Installation

You can follow the official installation tutorial of [SlowFast](https://github.com/facebookresearch/SlowFast) and [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs).
Or you can use the codes that I have configured for one-click installation [wdf_install_slowfast.sh](wdf_install_slowfast.sh)(**recommand**).

Here I list the directory structure of my enviorment, you can either place files in my format or modify the installation script yourself.


```bash
$ tree -L 1 /data1     # my root directory
/data1
├── config_slowfast    # packages used to install slowfast
├── Efficient-3DCNNs   # efficient 3d baseline
└── SlowFast_vis_0709  # slowfast main libary
```

```bash
$ tree -L 2 /data1/SlowFast_vis_0709/    # root directory of the SlowFast
/data1/SlowFast_vis_0709/
├── SlowFast
    ├── build
    ├── CODE_OF_CONDUCT.md
    ├── configs							# configs of each model, include Jester and Kinetics
    ├── CONTRIBUTING.md
    ├── demo							# video demo, 1) input a video, 2) select a model, 3) predict and output a result video
    ├── GETTING_STARTED.md
    ├── INSTALL.md						# official install tutorial
    ├── LICENSE
    ├── linter.sh
    ├── MODEL_ZOO.md
    ├── projects
    ├── README.md
    ├── setup.cfg
    ├── setup.py
    ├── slowfast						# main code
    ├── slowfast.egg-info
    ├── tools
    ├── wdf_all_run_scripts				# scripts used to train on AI-PLATFORM
    ├── wdf_install_slowfast.sh			# wdf's install script (recommand)
    └── wdf_visualization				# grad-cam visiualization

```

```bash
$tree -L 1 /data1/Efficient-3DCNNs/     # root directionary of the Efficient-3D (baseline)
..
├── annotation_Jester					
├── annotation_Kinetics					# here I provide the annotations of kinetics-400(not kinetics-600)
├── annotation_UCF101
├── calculate_FLOP.py
├── dataset.py
├── datasets
├── LICENSE
├── main.py
├── mean.py
├── model.py
├── models
├── opts.py
├── __pycache__
├── README.md
├── results-mobilenetv2-w1				# wdf trained models on kinetics-400
├── results-shufflenetv2-w025			# wdf trained models on kinetics-400
├── results-shufflenet-w2				# wdf trained models on kinetics-400
├── results-shufflev2-w1				# wdf trained models on kinetics-400
├── results-shufflev2-w2				# wdf trained models on kinetics-400
├── run-jester.sh
├── run-kinetics.sh
├── script								# scripts used to train(recommand to read)
├── spatial_transforms.py
├── speed_gpu.py
├── target_transforms.py
├── temporal_transforms.py
├── test_models.py
├── test.py
├── thop
├── train.py
├── utils
├── utils.py
└── validation.py
```



Then just run:

```bash
$ bash wdf_install_slowfast.sh
```



## Supproted Models and Pretrained Checkpoints

| Model Name            | Hyper-Parameters                                  | Checkpoints |
| --------------------- | ------------------------------------------------- | ----- |
| SlowFastDualAttention | Same as SlowFast, including ALPHA, BETA_INV, etc. | [BaiduYun(Password: kqqd)](https://pan.baidu.com/s/1k5tuqXz_4QQibgLHWm5muQ)|
| SlowFastShuffleNet    | Width=[1.0, 1.5, 2.0] , Groups=[1, 3]             |       |
| SlowFastShuffleNetV2  | Width=[0.25, 0.5, 1.0, 1.5, 2.0]                  |       |
| SlowFastMoibleNetV2   | Width=[0.5, 0.7, 1.0, 2.0]                        |       |
| SlowFastGhostNet      | Width=[1.0, 1.5, 2.0]                             |       |



## Supproted Datasets

1. Video classification:
   1. Kinetics-400 
   2. Jester-20bn-v1
2. Video detection: (Untested, becase we do not have these datasets )
   1. AVA  
   2. Charades



## Pipeline of SlowFast Networks

1. bulid model
2. prepare dataset (specify the video paths and labels, no need to extract and store frames in advance.)
3. train or test



## Quick Start

### Train 

To train our efficient dual-attention SlowFast networks, take **SlowFastShuffleNetV2** as an example,  you only need provide a config YAML file:

```bash
/data1/SlowFast_vis_0709/SlowFast$ python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_SHUFFLENETV2_8x8_R50_stepwise_multigrid.yaml 
```

The config YAML file specifies all hyper-parameters.



### Test

```bash
/data1/SlowFast_vis_0709/SlowFast$ python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_SHUFFLENETV2_8x8_R50_stepwise_multigrid.yaml \
TRAIN.ENABLE False \
TEST.ENABLE True \
TEST.CHECKPOINT_FILE_PATH /path/to/your/pretrained_model.pth
```



### Visualize (Grad-CAM)

We follow the concept of Grad-CAM to visualize the salient map of the SlowFast Networks.  `wdf_visualization` contains the codes.

**Usage:**

```bash
wdf_visualization$ python gradcam_video.py --help
usage: gradcam_video.py [-h] [--root_path ROOT_PATH] [--video_path VIDEO_PATH]
                        [--target_layer {s4_fuse,s5,s6,s5_fuse,s6_fuse,s8}]
                        [--yaml_cfg YAML_CFG]
                        [--checkpoint_pth CHECKPOINT_PTH]

Configs of Grad-CAM visualization.

optional arguments:
  -h, --help            show this help message and exit
  --root_path ROOT_PATH
                        root path of video
  --video_path VIDEO_PATH
                        video path
  --target_layer {s4_fuse,s5,s6,s5_fuse,s6_fuse,s8}
                        specify the layer to visualization, it should be the
                        last layer name
  --yaml_cfg YAML_CFG   yaml cfg file path
  --checkpoint_pth CHECKPOINT_PTH
                        checkpoint file path
```



**For example:**

```bash
python gradcam_video.py  \
	--yaml_cfg ../configs/Jester/SLOWFAST_MOBILENETV2_8x8_R50_stepwise_multigrid.yaml \
	--checkpoint_pth /data1/ADAS/Jester_SlowFastMoibleNetV2_W1/checkpoints/checkpoint_epoch_00100.pyth \
    --target_layer 's8' \
    --root_path /root/ \
    --video_path 20376.mp4  
```

| model_name   | target_layer |
| ------------ | ------------ |
| ghostnet     | s5           |
| mobilenetv2  | s8           |
| shufflenet   | s4_fuse      |
| shufflenetv2 | s4_fuse      |
| dual         | s5           |





### Demo (Video in, video out)

Here you can specify the pretrained model and video path to predict and grenerate a result.mp4 video.  It will output the FPS of this model.

1. set `TRAIN.ENABLE False` and ` TEST.ENABLE False`
2. spticify `DEMO.DATA_SOURCE` to the input video path, like `/data/my_video.mp4`.
3. specitfy `DEMO.OUTPUT_FILE` to `""` to display the predicted video, or `/path/to/result.mp4` to save the result video in mp4 format.

```bash
 python tools/run_net.py --cfg configs/Kinetics/C2D_8x8_R50.yaml TRAIN.ENABLE False TEST.ENABLE False TRAIN.CHECKPOINT_FILE_PATH /data1/SlowFast/checkpoints/checkpoint_epoch_00050.pyth
 
 # SHUFFLENET W2 G3
 python tools/run_net.py --cfg demo/Jester/SLOWFAST_SHUFFLENET_8x8_R50_stepwise_multigrid.yaml TEST.CHECKPOINT_FILE_PATH  "/data1/ADAS/JESTER_SlowFastShuffle_W2_G3/checkpoints/checkpoint_epoch_00100.pyth"   DEMO.OUTPUT_FILE /root/fps_result.mp4
 
 # SHUFFLENETV2 W2
 python tools/run_net.py --cfg demo/Jester/SLOWFAST_SHUFFLENETV2_8x8_R50_stepwise_multigrid.yaml TEST.CHECKPOINT_FILE_PATH  "/data1/ADAS/JESTER_SlowFastShuffleV2_W2/checkpoints/checkpoint_epoch_00100.pyth"   DEMO.OUTPUT_FILE /root/fps_result.mp4
 
 # MOBILENETV2 W1
 python tools/run_net.py --cfg demo/Jester/SLOWFAST_MOBILENETV2_8x8_R50_stepwise_multigrid.yaml TEST.CHECKPOINT_FILE_PATH  "/data1/ADAS/Jester_SlowFastMoibleNetV2_W1_New/checkpoints/checkpoint_epoch_00080.pyth"   DEMO.OUTPUT_FILE /root/fps_result.mp4
 
 # GHOSTNET W1
 python tools/run_net.py --cfg demo/Jester/SLOWFAST_GHOSTNET_8x8_R50_stepwise_multigrid.yaml TEST.CHECKPOINT_FILE_PATH  "/data1/ADAS/Jester_SlowFastGhostNet_W1/checkpoints/checkpoint_epoch_00023.pyth"   DEMO.OUTPUT_FILE /root/fps_result.mp4
```





## Pretrained models

**Kinetics-400**

| Model Name           | Hyper-Parameters     | Acc (and baseline) | Download |      |
| -------------------- | -------------------- | ------------------ | -------- | ---- |
| SlowFastShuffleNetV2 | Width=0.25           | 28.79 (24.11)      |          |      |
| SlowFastShuffleNetV2 | Width=1.0            | 38.54 (47.26)      |          |      |
| SlowFastShuffleNetV2 | Width=2.0            | 48.00 (54.22)      |          |      |
| SlowFastShuffleNet   | Width=2.0 , Groups=3 | 53.84 (51.06)      |          |      |
| SlowFastShuffleNet   | Width=2.0 , Groups=1 | 54.99 (50.19)      |          |      |
| SlowFastMoibleNetV2  | Width=1.0            | 48.12 (38.54)      |          |      |
| SlowFastGhostNet     | Width=1.0            | 46.03              |          |      |



**Jester-20BN**

| Model Name           | Hyper-Parameters     | Acc (and baseline) | Download |      |
| -------------------- | -------------------- | ------------------ | -------- | ---- |
| SlowFastShuffleNetV2 | Width=2.0            | (93.71)            |          |      |
| SlowFastShuffleNet   | Width=2.0 , Groups=3 | (93.54)            |          |      |
| SlowFastMoibleNetV2  | Width=1.0            | (94.59)            |          |      |
