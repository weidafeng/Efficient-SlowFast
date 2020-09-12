"""
Modified by wdf base on github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
import time
import argparse
from misc_functions import (save_class_activation_images,
                            save_class_activation_images_as_gif,
                            normalization)


def myargparse():
    parser = argparse.ArgumentParser(
        description='Configs of Grad-CAM visualization.')

    # input video
    parser.add_argument('--root_path', type=str,
                        default="/dataset/guojietian/kinetics400/",
                        help='root path of video')
    parser.add_argument('--video_path', type=str,
                        # required=True,
                        default="video/val_256/6qHa-mSVa1U_000064_000074.mp4",
                        help='video path')

    # target layer to visualize
    parser.add_argument('--target_layer', type=str,
                        # required=True,
                        default='s4_fuse',
                        choices=['s4_fuse', 's5', 's6', 's5_fuse', 's6_fuse',
                                 's8'],
                        help='specify the layer to visualization, it should be the last layer name')
    # target label to visualize
    parser.add_argument('--target_label', type=str,
                        # required=True,
                        default=None,
                        help='specify the class to visualization')

    # cfg and pth file
    parser.add_argument('--yaml_cfg', type=str,
                        # required=True,
                        default="/data1/SlowFast_vis_0709/SlowFast/configs/Kinetics/SLOWFAST_SHUFFLENET_8x8_R50_stepwise_multigrid.yaml",
                        help='yaml cfg file path')
    parser.add_argument('--checkpoint_pth', type=str,
                        # required=True,
                        default="/data1/ADAS/KINETICS_SlowFastShuffleNet_W1_G3/checkpoints/checkpoint_epoch_00008.pyth",
                        help='checkpoint file path')

    parser.add_argument('--print_flops',
                        action='store_true',
                        default=False,
                        help="print parameters and flops of this model")

    args = parser.parse_args()
    return args


class CamVideoExtractor():
    """
        Extracts cam features from the SlowFast model
        target_layer:
            s1
            s2
            s2_fuse
            s3
            s3_fuse
            s4
            s4_fuse
            s5
            head

    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients_slow = None
        self.gradients_fast = None

    def save_gradient_slow(self, grad):
        self.gradients_slow = grad

    def save_gradient_fast(self, grad):
        self.gradients_fast = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_name, module in self.model._modules.items():
            if module_name == 'head':
                # self.head = module
                break
            if module_name == "pathway0_pool":
                x[0] = module(x[0])
            elif module_name == "pathway1_pool":
                x[1] = module(x[1])
            else:
                x = module(x)  # Forward
            if module_name == self.target_layer:
                x[0].register_hook(self.save_gradient_slow)
                x[1].register_hook(self.save_gradient_fast)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.head(x)
        return conv_output, x


class GradVideoCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamVideoExtractor(self.model, target_layer)

    def generate_cam_videos(self, input_image, target_class=None,
                            original_image=None,
                            total_images=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        print("Conv_output.shape: ", conv_output[0].shape, conv_output[1].shape)
        print("Model_output.shape: ", model_output.shape)
        print("Original_image.shape: ", original_image[0].shape,
              original_image[1].shape)

        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][int(target_class)] = 1

        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target

        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # ________slow path___________
        print("*" * 10, end=' ')
        print("Slow path  ", "*" * 10)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients_slow.data.numpy()[0]
        # Get convolution outputs
        target = conv_output[0].data.numpy()[0]

        slow_cams = []
        for idx in range(guided_gradients.shape[1]):
            # Get weights from gradients
            weights = np.mean(guided_gradients[:, idx, :, :],
                              axis=(1, 2))  # Take averages for each gradient

            # Create empty numpy array for cam
            cam = np.ones(target.shape[2:], dtype=np.float32)

            # Multiply each weight with its conv output and then, sum
            for i, w in enumerate(weights):
                cam += w * np.mean(target, 1)[i, :, :]

            cam = np.maximum(cam, 0)
            cam = normalization(cam)  # min-max

            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            print("cam.shape", cam.shape)

            cam_slow = np.uint8(Image.fromarray(np.uint8(cam)).resize(
                (original_image[0].size(2),
                 original_image[0].size(3)),
                Image.ANTIALIAS)) / 255

            slow_cams.append(cam_slow)

        # ________fast path___________
        print("*" * 10, end=' ')
        print("Fast path  ", "*" * 10)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients_fast.data.numpy()[0]
        # Get convolution outputs
        target = conv_output[1].data.numpy()[0]
        fast_cams = []
        for idx in range(guided_gradients.shape[1]):
            # Get weights from gradients
            weights = np.mean(guided_gradients[:, idx, :, :],
                              axis=(1, 2))  # Take averages for each gradient
            # Create empty numpy array for cam
            cam = np.ones(target.shape[2:], dtype=np.float32)
            print("cam.shape", cam.shape)

            # Multiply each weight with its conv output and then, sum
            for i, w in enumerate(weights):
                cam += w * np.mean(target, 1)[i, :, :]
            cam = np.maximum(cam, 0)
            cam = normalization(cam)  # min-max

            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize

            if total_images is not None:
                if idx >= total_images:
                    idx = total_images - 1

            cam_fast = np.uint8(Image.fromarray(np.uint8(cam)).resize(
                (original_image[1].size(2),
                 original_image[1].size(3)),
                Image.ANTIALIAS)) / 255

            fast_cams.append(cam_fast)

        return slow_cams, fast_cams


def _build_and_load_model(yaml_cfg, checkpoint_pth):
    from slowfast.config.defaults import get_cfg
    from slowfast.models import build_model
    import slowfast.utils.checkpoint as cu

    # load configs
    cfg = get_cfg()
    cfg.merge_from_file(yaml_cfg)

    # You can also modify params here, like:
    # cfg.MODEL.MODEL_NAME = 'SlowFastGhostNet'
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_pth
    cfg.NUM_GPUS = 1
    # cfg.TRAIN.CHECKPOINT_FILE_PATH = checkpoint_pth

    # build model
    model = build_model(cfg).cpu()
    cu.load_test_checkpoint(cfg, model)
    return model, cfg


def _prepare_video_dataset(cfg, path_to_vid):
    from slowfast.datasets import video_container as container
    from slowfast.datasets import decoder as decoder
    from slowfast.datasets import utils as utils

    video_container = container.get_video_container(
        path_to_vid,
        cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
        cfg.DATA.DECODING_BACKEND,
    )
    sampling_rate = utils.get_random_sampling_rate(
        cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
        cfg.DATA.SAMPLING_RATE,
    )

    # Decode video. Meta info is used to perform selective decoding.
    frames = decoder.decode(
        video_container,
        sampling_rate,
        cfg.DATA.NUM_FRAMES,
        -1,  # random sampling
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
        target_fps=cfg.DATA.TARGET_FPS,
        backend=cfg.DATA.DECODING_BACKEND,
    )

    # Perform color normalization.
    frames = utils.tensor_normalize(
        frames, cfg.DATA.MEAN, cfg.DATA.STD
    )
    # T H W C -> C T H W.
    frames = frames.permute(3, 0, 1, 2)
    # Perform data augmentation.
    frames = utils.spatial_sampling(
        frames,
        spatial_idx=1,  # [0, 1, 2] means [left, center, right]
        min_scale=cfg.DATA.TEST_CROP_SIZE,
        max_scale=cfg.DATA.TEST_CROP_SIZE,
        crop_size=cfg.DATA.TEST_CROP_SIZE,
        random_horizontal_flip=cfg.DATA.RANDOM_FLIP,
        inverse_uniform_sampling=cfg.DATA.INV_UNIFORM_SAMPLE,
    )
    origin_frames = frames.clone()
    origin_frames = origin_frames.permute(1, 2, 3, 0)
    origin_frames = utils.revert_tensor_normalize(
        origin_frames, cfg.DATA.MEAN, cfg.DATA.STD
    )
    origin_frames = origin_frames.permute(3, 0, 1, 2)

    origin_frames = [origin_frames[:, ::cfg.SLOWFAST.ALPHA, ...], origin_frames]

    frames = utils.pack_pathway_output(cfg, frames)

    frames = [frame.unsqueeze(0) for frame in frames]

    print("*" * 10, end=' ')
    print("Input shape  ", "*" * 10)
    print(frames[0].shape, frames[1].shape)

    return origin_frames, frames


def log_model_flops_per_layer(model, inputs):
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model,
                                             inputs,
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def plot_grad_cams():
    args = myargparse()
    # 1. video path
    root_path = args.root_path
    path_to_vid = root_path + args.video_path

    # 2. target layer to visualize
    target_layer = args.target_layer  # 's4_fuse'
    target_label = args.target_label  # none means the predicted label, otherwise you can specify int label

    # 3.
    yaml_cfg = args.yaml_cfg
    checkpoint_pth = args.checkpoint_pth

    # # official baseline 1
    # yaml_cfg = "/data1/SlowFast_vis_0709/SlowFast/configs/Kinetics/c2/SLOWFAST_8x8_R50.yaml"
    # checkpoint_pth = "/data1/ADAS/KINETICS_BASELINE/SLOWFAST_8x8_R50.pkl"
    # # official baseline 2
    # yaml_cfg = "/data1/SlowFast_vis_0709/SlowFast/configs/Kinetics/c2/SLOWFAST_4x16_R50.yaml"
    # checkpoint_pth = "/data1/ADAS/KINETICS_BASELINE/SLOWFAST_4x16_R50.pkl"

    # 4. save path
    save_path = yaml_cfg.split("/")[-1].split("_")[1] + "_" + \
                path_to_vid.split('/')[-1]

    # build and load pre-trained model
    model, cfg = _build_and_load_model(yaml_cfg=yaml_cfg,
                                       checkpoint_pth=checkpoint_pth)

    # load video and pre-process
    origin_frames, frames = _prepare_video_dataset(cfg,
                                                   path_to_vid)

    if args.print_flops:
        # -- print parameters and flops of this model --
        input_shape = ((frames[0].size(1), frames[0].size(2), frames[0].size(3),
                        frames[0].size(4)),
                       (frames[1].size(1), frames[1].size(2), frames[1].size(3),
                        frames[1].size(4)),)
        print("Input Shape: ", input_shape)
        log_model_flops_per_layer(model, input_shape)

    # Grad cam
    grad_cam = GradVideoCam(model, target_layer)
    # Generate cam mask
    slow_cams, fast_cams = grad_cam.generate_cam_videos(frames,
                                                        target_label,
                                                        origin_frames,
                                                        )
    # Save mask
    for idx in range(len(slow_cams)):
        save_file_name = str(save_path) + "_" + str(target_label) + "_" + str(
            idx) + "_slow_"
        save_class_activation_images(origin_frames[0][:, idx, :, :],
                                     slow_cams[idx],
                                     save_file_name,
                                     is_video=True)
    for idx in range(len(fast_cams)):
        save_file_name = str(save_path) + "_" + str(target_label) + "_" + str(
            idx) + "_fast_"
        save_class_activation_images(origin_frames[1][:, idx, :, :],
                                     fast_cams[idx],
                                     save_file_name,
                                     is_video=True)
    # save gif
    save_class_activation_images_as_gif(origin_frames[0],
                                        slow_cams,
                                        str(save_path) + "_" + str(
                                            target_label) + "_slow",
                                        is_video=True)
    save_class_activation_images_as_gif(origin_frames[1],
                                        fast_cams,
                                        str(save_path) + "_" + str(
                                            target_label) + "_fast",
                                        is_video=True)
    print("*" * 10, end=' ')
    print("Grad cam completed  ", "*" * 10)


if __name__ == '__main__':
    # plot_grad_cam_for_kinetics_video()
    plot_grad_cams()
