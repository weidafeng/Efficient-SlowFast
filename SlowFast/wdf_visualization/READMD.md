
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

| model_name   | target_layer            |
| ------------ | ----------------------- |
| ghostnet     | s5                      |
| mobilenetv2  | s8                      |
| shufflenet   | s4_fuse                 |
| shufflenetv2 | s4_fuse                 |
| dual         | s5                      |

