#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import random
import torch
import torchvision.io as io
import os
from PIL import Image
import random
from torchvision.transforms import ToTensor
from . import transform
from slowfast.datasets import utils


class ToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)


# def fix_compose_transform(transform):
#         if isinstance(transform.transforms[-1], torchvision.transforms.ToTensor):
#             transform = torchvision.transforms.Compose([
#                 # *transform.transforms[:-1],
#                 ToNumpy(),
#                 torchvision.transforms.ToTensor()
#             ])
#         return transform
#
# ToTensor = fix_compose_transform(torchvision.transforms.Compose([torchvision.transforms.ToTensor]))
# print(ToTensor, type(ToTensor))


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def get_start_end_idx_in_the_middle(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    输入视频的总帧数、需要截取的图像帧数（考虑了采样率）、训练集还是测试集（由clip_idx控制，如果是-1，表示训练集），以及测试阶段一共需要截取几个clips
    1. 选取起始帧：
        如果当前文件夹内图像不足指定帧数，则从第一帧开始；
        否则从前面一小段随机选取起始帧（保证需要的图像都在文件夹内）
    2. 计算结束帧
        如果图像足够多，则默认为：结束帧 = 起始帧 + 待截取帧数 - 1
        如果图像不足，则： 结束帧 = 文件夹内总帧数 - 1
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = min(start_idx + clip_size - 1, video_size - 1)  # wdf-fix
    return int(start_idx), int(end_idx)


def get_start_end_idx_in_the_middle_fix0710(video_size, clip_size, clip_idx,
                                            num_clips, drop_ratio=0.1):
    """
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
        sampling_rate: 采样率，整数
    """
    # 只有1帧
    if video_size == 1:
        return 0, 0, 1
    # 去掉前10%和最后10%
    strip_out = int(drop_ratio * video_size)
    video_size_available = video_size - strip_out * 2

    # 计算采样间隔
    sampling_rate = video_size_available // clip_size
    sampling_rate = 1 if sampling_rate == 0 else sampling_rate
    # 随机选取起始帧
    start_idx = random.randint(strip_out, strip_out + sampling_rate)
    end_idx = min(start_idx + sampling_rate * clip_size,
                  video_size - strip_out - 1)

    return int(start_idx), int(end_idx), sampling_rate


def pyav_decode_stream(
        container, start_pts, end_pts, stream, stream_name, buffer_size=0
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts


def torchvision_decode(
        video_handle,
        sampling_rate,
        num_frames,
        clip_idx,
        video_meta,
        num_clips=10,
        target_fps=30,
        modalities=("visual",),
        max_spatial_scale=0,
):
    """
    If video_meta is not empty, perform temporal selective decoding to sample a
    clip from the video with TorchVision decoder. If video_meta is empty, decode
    the entire video and update the video_meta.
    Args:
        video_handle (bytes): raw bytes of the video file.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the clip_idx-th video clip.
        video_meta (dict): a dict contains VideoMetaData. Details can be found
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps.
        modalities (tuple): tuple of modalities to decode. Currently only
            support `visual`, planning to support `acoustic` soon.
        max_spatial_scale (int): the maximal resolution of the spatial shorter
            edge size during decoding.
    Returns:
        frames (tensor): decoded frames from the video.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): if True, the entire video was decoded.
    """
    # Convert the bytes to a tensor.
    video_tensor = torch.from_numpy(np.frombuffer(video_handle, dtype=np.uint8))

    decode_all_video = True
    video_start_pts, video_end_pts = 0, -1
    # The video_meta is empty, fetch the meta data from the raw video.
    if len(video_meta) == 0:
        # Tracking the meta info for selective decoding in the future.
        meta = io._probe_video_from_memory(video_tensor)
        # Using the information from video_meta to perform selective decoding.
        video_meta["video_timebase"] = meta.video_timebase
        video_meta["video_numerator"] = meta.video_timebase.numerator
        video_meta["video_denominator"] = meta.video_timebase.denominator
        video_meta["has_video"] = meta.has_video
        video_meta["video_duration"] = meta.video_duration
        video_meta["video_fps"] = meta.video_fps
        video_meta["audio_timebas"] = meta.audio_timebase
        video_meta["audio_numerator"] = meta.audio_timebase.numerator
        video_meta["audio_denominator"] = meta.audio_timebase.denominator
        video_meta["has_audio"] = meta.has_audio
        video_meta["audio_duration"] = meta.audio_duration
        video_meta["audio_sample_rate"] = meta.audio_sample_rate

    if (
            video_meta["has_video"]
            and video_meta["video_denominator"] > 0
            and video_meta["video_duration"] > 0
    ):
        decode_all_video = False
        start_idx, end_idx = get_start_end_idx(
            video_meta["video_fps"] * video_meta["video_duration"],
            sampling_rate * num_frames / target_fps * video_meta["video_fps"],
            clip_idx,
            num_clips,
        )
        # Convert frame index to pts.
        pts_per_frame = (
                video_meta["video_denominator"] / video_meta["video_fps"]
        )
        video_start_pts = int(start_idx * pts_per_frame)
        video_end_pts = int(end_idx * pts_per_frame)

    # Decode the raw video with the tv decoder.
    v_frames, _ = io._read_video_from_memory(
        video_tensor,
        seek_frame_margin=1.0,
        read_video_stream="visual" in modalities,
        video_width=0,
        video_height=0,
        video_min_dimension=max_spatial_scale,
        video_pts_range=(video_start_pts, video_end_pts),
        video_timebase_numerator=video_meta["video_numerator"],
        video_timebase_denominator=video_meta["video_denominator"],
    )
    return v_frames, video_meta["video_fps"], decode_all_video


def pyav_decode(
        container, sampling_rate, num_frames, clip_idx, num_clips=10,
        target_fps=30
):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.

    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    # Try to fetch the decoding information from the video head. Some of the
    # videos does not support fetching the decoding information, for that case
    # it will get None duration.
    fps = float(container.streams.video[0].average_rate)
    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration

    if duration is None:
        # If failed to fetch the decoding information, decode the entire video.
        decode_all_video = True
        video_start_pts, video_end_pts = 0, math.inf
    else:
        # Perform selective decoding.
        decode_all_video = False
        start_idx, end_idx = get_start_end_idx(
            frames_length,
            sampling_rate * num_frames / target_fps * fps,
            clip_idx,
            num_clips,
        )
        timebase = duration / frames_length
        video_start_pts = int(start_idx * timebase)
        video_end_pts = int(end_idx * timebase)

    frames = None
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        video_frames, max_pts = pyav_decode_stream(
            container,
            video_start_pts,
            video_end_pts,
            container.streams.video[0],
            {"video": 0},
        )
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))
    return frames, fps, decode_all_video


def decode(
        container,
        sampling_rate,
        num_frames,
        clip_idx=-1,
        num_clips=10,
        video_meta=None,
        target_fps=30,
        backend="pyav",
        max_spatial_scale=0,
        jester=False,  # wdf add
        jester_test=True  # wdf add
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
        jester (bool): False,  wdf add for the jester dataset
        jester_test (bool): True, use to decide whether to do color jittering.
    Returns:
        frames (tensor): decoded frames from the video.
        # 示例：
        # frames = decode(container, 8, 12, -1, 10)
        # frame shape: torch.Size([12, 256, 454, 3])

    # 关于VideoMetaData()
    https://gitee.com/nustart/torchvision/blob/master/torchvision/io/_video_opt.py#L53
    def __init__(self):
        self.has_video = False
        self.video_timebase = Timebase(0, 1)
        self.video_duration = 0.0
        self.video_fps = 0.0
        self.has_audio = False
        self.audio_timebase = Timebase(0, 1)
        self.audio_duration = 0.0
        self.audio_sample_rate = 0.0
    """
    # Currently support two decoders: 1) PyAV, and 2) TorchVision.
    assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
    try:
        if backend == "pyav":
            frames, fps, decode_all_video = pyav_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                num_clips,
                target_fps,
            )
        elif backend == "torchvision":
            frames, fps, decode_all_video = torchvision_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                video_meta,
                num_clips,
                target_fps,
                ("visual",),
                max_spatial_scale,
            )
        else:
            raise NotImplementedError(
                "Unknown decoding backend {}".format(backend)
            )
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        return None

    # Return None if the frames was not decoded successfully.
    if frames is None or frames.size(0) == 0:
        return None
    start_idx, end_idx = get_start_end_idx(
        frames.shape[0],
        num_frames * sampling_rate * fps / target_fps,
        clip_idx if decode_all_video else 0,
        num_clips if decode_all_video else 1,
    )
    # Perform temporal sampling from the decoded video.
    frames = temporal_sampling(frames, start_idx, end_idx, num_frames)

    # for the jester [train, val] dataset, we do random color jittering.
    if jester and not jester_test:
        bright = random.uniform(0.4, 1.4)
        contrast = random.uniform(0.4, 1.4)
        color = random.uniform(0.4, 1.4)
        frames = frames.permute(0, 3, 1, 2)  # [h,h,w,c] --> [b, c, h, w]
        # transform.save_tensor(frames, p='/root/raw{}.gif'.format(color))
        frames = transform.RandomColorJitter(bright=bright,
                                   contrast=contrast,
                                   color=color)(frames)

        frames = torch.as_tensor(np.stack(frames))
        # transform.save_tensor(frames, p='/root/raw_result{}.gif'.format(color))
        frames = frames.permute(0, 2, 3, 1)
    return frames





def wheel_decoder(
        path_to_video,
        sampling_rate,
        num_frames,
        clip_idx,
        num_clips,
        # min_scale,
        # max_scale,
        target_scale,
        phase,
        half_face=False
):
    '''

    :param path_to_video:
    :param sampling_rate: self.cfg.DATA.SAMPLING_RATE, 每隔几帧取一帧。如果文件夹内图像很多的话，就间隔取；如果图像不够指定帧，则不间隔
    :param num_frames: self.cfg.DATA.NUM_FRAMES, Fast分支需要的帧数
    :param clip_idx: temporal_sample_index,
    :param num_clips: self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
    :param target_fps: self.cfg.DATA.TARGET_FPS,（图像数据集不需要这个参数）
    :param fps: 30，与target_fps一致（图像数据集不需要这个参数）
    :return:

    frames = decoder.wheel_decoder(
        self._path_to_videos[index],
        sampling_rate=self.cfg.DATA.SAMPLING_RATE,
        num_frames=self.cfg.DATA.NUM_FRAMES,
        clip_idx=temporal_sample_index,
        num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
        target_fps=self.cfg.DATA.TARGET_FPS,
        min_scale=min_scale,
        max_scale=max_scale
    )

    '''
    # 输入文件夹路径，读取文件夹内的所有图片，并执行时间采样
    images = os.listdir(path_to_video)
    frame_count = len(images)  # 这段视频的总帧数

    # 每隔几帧取一帧。如果文件夹内图像很多的话，就间隔取；如果图像不够指定帧，则不间隔
    if frame_count < num_frames * sampling_rate:
        sampling_rate = 1

    # 先读取一帧，得到当前文件夹内图像的大致宽高信息
    img = Image.open(os.path.join(path_to_video, images[0]))
    frame_width = int(img.size[0])
    frame_height = int(img.size[1])
    # 如果是测试阶段，或者是灰度图，则不需要进行图像增广（亮度、对比度、色调、饱和度等等），只做空间随机裁剪
    if phase == 'test' or len(img.split()) < 3:
        DO_COLOR_AUGUMENT = False
    else:
        DO_COLOR_AUGUMENT = True
        COLOR_AUGUMENT = transform.Compose([
            # transform.RandomResize(),
            transform.RandomRotate(),
            # transform.Gaussian_blur(),
            transform.SaltImage(),
        ])

    # 存储当前文件夹下包含的所有图像
    # 注意，ADAS数据集里，即便同一个文件夹内的图像大小也不一致，需要先做resize

    # 随机缩放：
    # 1. 从左上角区域随机选取起始点
    # 2. 从左上角开始，截取到右下角的全部区域
    # 3. 把截取到的图像resize成指定大小
    start_width = np.random.randint(0, 0.1 * frame_width)
    start_height = np.random.randint(0, 0.1 * frame_height)

    # 计算需要采样的帧的索引（只读取用得到的帧，可以节省大量的时间）
    start_idx, end_idx = get_start_end_idx_in_the_middle(
        frame_count,
        num_frames * sampling_rate,
        clip_idx,
        num_clips,
    )
    frames = []
    for idx, img in enumerate(images):
        # 只读取需要的帧
        if idx < start_idx:
            continue
        if idx > end_idx:
            break
        if idx % sampling_rate == 0:
            if not half_face:
                img = Image.open(os.path.join(path_to_video, img)).crop(
                    (start_width, start_height, frame_width, frame_height))
            elif half_face:
                r = (0.6 - 0.5) * np.random.random() + 0.5  # 在0.5和0.6范围内选一个比例
                img = Image.open(os.path.join(path_to_video, img)).crop(
                    (start_width, start_height, frame_width, r * frame_height))
            img = img.resize((target_scale, target_scale))  # 按要求缩放
            if DO_COLOR_AUGUMENT:
                img = COLOR_AUGUMENT(img)  # 随机、旋转、椒盐噪声等
            # img_tensor = ToTensor()(img)  # raw version
            img_tensor = ToTensor()(
                ToNumpy()(img))  # 修改numpy array not writable错误
            frames.append(img_tensor)

    # print(1, len(frames), path_to_video)
    frames = torch.as_tensor(np.stack(frames))
    # 只对彩色图做图像增广（亮度、对比度、色调、饱和度等等），只做空间随机裁剪
    if DO_COLOR_AUGUMENT:
        frames = transform.color_jitter(
            frames,  # [b, c, h, w]
            img_brightness=0.4,
            img_contrast=0.4,
            img_saturation=0.4,
            mode='RGB'
        )
    # print(type(frames), frames.shape)
    # tensor [100, 3, 320, 300]
    # 然后对当前文件夹下所有图像做时序采样(如果不够帧数，则循环复制）
    # frames = transform.TemporalRandomCrop(size=num_frames,
    #                                       downsample=sampling_rate)(frames)

    # 如果帧数不够，则复制
    if frames.shape[0] != num_frames:
        index = torch.linspace(0, frames.shape[0], num_frames)
        index = torch.clamp(index, 0, frames.shape[0] - 1).long()
        frames = torch.index_select(frames, 0, index)

    # 随机水平翻转(GOP为单位）
    frames, _ = transform.horizontal_flip(0.5, frames)
    # print(2, len(frames), path_to_video)

    # print(type(frames), frames.shape)
    # tensor [32, 3, 320, 300]
    return frames


def wheel_decoder_gray_style(
        path_to_video,
        sampling_rate,
        num_frames,
        clip_idx,
        num_clips,
        # min_scale,
        # max_scale,
        target_scale,
        phase,
        half_face=False
):
    '''
    :param path_to_video:
    :param sampling_rate: self.cfg.DATA.SAMPLING_RATE, 每隔几帧取一帧。如果文件夹内图像很多的话，就间隔取；如果图像不够指定帧，则不间隔
    :param num_frames: self.cfg.DATA.NUM_FRAMES, Fast分支需要的帧数
    :param clip_idx: temporal_sample_index,
    :param num_clips: self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
    :param target_fps: self.cfg.DATA.TARGET_FPS,（图像数据集不需要这个参数）
    :param fps: 30，与target_fps一致（图像数据集不需要这个参数）
    :return:

    frames = decoder.wheel_decoder(
        self._path_to_videos[index],
        sampling_rate=self.cfg.DATA.SAMPLING_RATE,
        num_frames=self.cfg.DATA.NUM_FRAMES,
        clip_idx=temporal_sample_index,
        num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
        target_fps=self.cfg.DATA.TARGET_FPS,
        min_scale=min_scale,
        max_scale=max_scale
    )

    '''
    # 输入文件夹路径，读取文件夹内的所有图片，并执行时间采样
    images = os.listdir(path_to_video)
    frame_count = len(images)  # 这段视频的总帧数

    # 每隔几帧取一帧。如果文件夹内图像很多的话，就间隔取；如果图像不够指定帧，则不间隔
    if frame_count < num_frames * sampling_rate:
        sampling_rate = 1

    # 先读取一帧，得到当前文件夹内图像的大致宽高信息
    img = images[0] if images[0].endswith('jpg') else images[-1]
    img = Image.open(os.path.join(path_to_video, img))
    frame_width = int(img.size[0])
    frame_height = int(img.size[1])
    # 如果是测试阶段，或者是灰度图，则不需要进行图像增广（亮度、对比度、色调、饱和度等等），只做空间随机裁剪
    if phase != 'train':
        DO_COLOR_AUGUMENT = False
    else:
        DO_COLOR_AUGUMENT = True
        COLOR_AUGUMENT = transform.Compose([
            # transform.RandomResize(),
            transform.RandomRotate(),
            # transform.Gaussian_blur(),
            transform.SaltImage(),
        ])

    # 存储当前文件夹下包含的所有图像
    # 注意，ADAS数据集里，即便同一个文件夹内的图像大小也不一致，需要先做resize

    # 随机缩放：
    # 1. 从左上角区域随机选取起始点
    # 2. 从左上角开始，截取到右下角的全部区域
    # 3. 把截取到的图像resize成指定大小
    start_width = np.random.randint(0, 0.1 * frame_width)
    start_height = np.random.randint(0, 0.1 * frame_height)

    # 计算需要采样的帧的索引（只读取用得到的帧，可以节省大量的时间）
    start_idx, end_idx, sampling_rate = get_start_end_idx_in_the_middle_fix0710(
        frame_count,
        num_frames,
        clip_idx,
        num_clips,
    )
    # print(start_idx, end_idx, sampling_rate)
    frames = []
    for idx, img in enumerate(images):
        # 只读取需要的帧
        if idx < start_idx:
            continue
        if idx > end_idx:
            break
        if idx % sampling_rate == 0 and img.endswith('jpg'):
            img = Image.open(os.path.join(path_to_video, img)).convert("L")
            if phase in ['train', 'val']:
                if not half_face:
                    img = img.crop(
                        (start_width, start_height, img.size[0], img.size[1]))
                elif half_face:
                    r = (
                                0.6 - 0.5) * np.random.random() + 0.5  # 在0.5和0.6范围内选一个比例
                    img = img.crop((start_width, start_height, img.size[0],
                                    r * img.size[1]))
            elif phase == 'test':
                if not half_face:
                    img = img.crop((0, 0, img.size[0], img.size[1]))
                elif half_face:
                    r = (
                                0.6 - 0.5) * np.random.random() + 0.5  # 在0.5和0.6范围内选一个比例
                    img = img.crop((0, 0, img.size[0], r * img.size[1]))

            img = img.resize((target_scale, target_scale))  # 按要求缩放
            if DO_COLOR_AUGUMENT:
                img = COLOR_AUGUMENT(img)  # 随机、旋转、椒盐噪声等
            # img_tensor = ToTensor()(img)  # raw version
            img_tensor = ToTensor()(
                ToNumpy()(img))  # 修改numpy array not writable错误
            frames.append(img_tensor)

    # print(1, len(frames), path_to_video)
    frames = torch.as_tensor(np.stack(frames))
    # 只对彩色图做图像增广（亮度、对比度、色调、饱和度等等），只做空间随机裁剪
    if DO_COLOR_AUGUMENT:
        frames = transform.color_jitter(
            frames,
            img_brightness=0.2,
            img_contrast=0,
            img_saturation=0,
            mode='L'
        )
    # print(type(frames), frames.shape)
    # tensor [100, 3, 320, 300]
    # 然后对当前文件夹下所有图像做时序采样(如果不够帧数，则循环复制）
    # frames = transform.TemporalRandomCrop(size=num_frames,
    #                                       downsample=sampling_rate)(frames)

    # 如果帧数不够，则复制
    if frames.shape[0] != num_frames:
        index = torch.linspace(0, frames.shape[0], num_frames)
        index = torch.clamp(index, 0, frames.shape[0] - 1).long()
        frames = torch.index_select(frames, 0, index)

    # 随机水平翻转(GOP为单位）
    if DO_COLOR_AUGUMENT:
        frames, _ = transform.horizontal_flip(0.5, frames)
    # print(2, len(frames), path_to_video)

    # print(type(frames), frames.shape)
    # tensor [32, 3, 320, 300]
    return frames


def smoke_decoder_gray_style(
        path_to_video,
        sampling_rate,
        num_frames,
        clip_idx,
        num_clips,
        # min_scale,
        # max_scale,
        target_scale,
        phase,
        half_face=False
):
    '''
    :param path_to_video:
    :param sampling_rate: self.cfg.DATA.SAMPLING_RATE, 每隔几帧取一帧。如果文件夹内图像很多的话，就间隔取；如果图像不够指定帧，则不间隔
    :param num_frames: self.cfg.DATA.NUM_FRAMES, Fast分支需要的帧数
    :param clip_idx: temporal_sample_index,
    :param num_clips: self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
    :param target_fps: self.cfg.DATA.TARGET_FPS,（图像数据集不需要这个参数）
    :param fps: 30，与target_fps一致（图像数据集不需要这个参数）
    :return:

    frames = decoder.wheel_decoder(
        self._path_to_videos[index],
        sampling_rate=self.cfg.DATA.SAMPLING_RATE,
        num_frames=self.cfg.DATA.NUM_FRAMES,
        clip_idx=temporal_sample_index,
        num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
        target_fps=self.cfg.DATA.TARGET_FPS,
        min_scale=min_scale,
        max_scale=max_scale
    )

    '''
    # 输入文件夹路径，读取文件夹内的所有图片，并执行时间采样
    images = os.listdir(path_to_video)
    frame_count = len(images)  # 这段视频的总帧数

    # 每隔几帧取一帧。如果文件夹内图像很多的话，就间隔取；如果图像不够指定帧，则不间隔
    if frame_count < num_frames * sampling_rate:
        sampling_rate = 1

    # 先读取一帧，得到当前文件夹内图像的大致宽高信息
    img = images[0] if images[0].endswith('jpg') else images[-1]
    img = Image.open(os.path.join(path_to_video, img))
    frame_width = int(img.size[0])
    frame_height = int(img.size[1])
    # 如果是测试阶段，或者是灰度图，则不需要进行图像增广（亮度、对比度、色调、饱和度等等），只做空间随机裁剪
    if phase != 'train':
        DO_COLOR_AUGUMENT = False
    else:
        DO_COLOR_AUGUMENT = True
        COLOR_AUGUMENT = transform.Compose([
            # transform.RandomResize(),
            transform.RandomRotate(),
            # transform.Gaussian_blur(),
            transform.SaltImage(),
        ])

    # 存储当前文件夹下包含的所有图像
    # 注意，ADAS数据集里，即便同一个文件夹内的图像大小也不一致，需要先做resize

    # 随机缩放：
    # 1. 从左上角区域随机选取起始点
    # 2. 从左上角开始，截取到右下角的全部区域
    # 3. 把截取到的图像resize成指定大小
    start_width = np.random.randint(0, 0.1 * frame_width)
    start_height = np.random.randint(0, 0.1 * frame_height)

    # 计算需要采样的帧的索引（只读取用得到的帧，可以节省大量的时间）
    start_idx, end_idx, _ = get_start_end_idx_in_the_middle_fix0710(
        frame_count,
        num_frames,
        clip_idx,
        num_clips,
        drop_ratio=0.2  # 去掉前20% 和 后20%
    )
    # print(start_idx, end_idx, sampling_rate)
    frames = []
    for idx, img in enumerate(images):
        # 只读取需要的帧
        if idx < start_idx:
            continue
        if idx > end_idx or len(frames) > num_frames:
            break
        if (idx - start_idx) % sampling_rate == 0 and img.endswith('jpg'):
            img = Image.open(os.path.join(path_to_video, img)).convert("L")
            if phase in ['train', 'val']:
                if not half_face:
                    img = img.crop(
                        (start_width, start_height, img.size[0], img.size[1]))
                elif half_face:
                    r = (
                                0.6 - 0.5) * np.random.random() + 0.5  # 在0.5和0.6范围内选一个比例
                    img = img.crop((start_width, start_height, img.size[0],
                                    r * img.size[1]))
            elif phase == 'test':
                if not half_face:
                    img = img.crop((0, 0, img.size[0], img.size[1]))
                elif half_face:
                    r = (
                                0.6 - 0.5) * np.random.random() + 0.5  # 在0.5和0.6范围内选一个比例
                    img = img.crop((0, 0, img.size[0], r * img.size[1]))

            img = img.resize((target_scale, target_scale))  # 按要求缩放
            if DO_COLOR_AUGUMENT:
                img = COLOR_AUGUMENT(img)  # 随机、旋转、椒盐噪声等
            # img_tensor = ToTensor()(img)  # raw version
            img_tensor = ToTensor()(
                ToNumpy()(img))  # 修改numpy array not writable错误
            frames.append(img_tensor)

    # print(1, len(frames), path_to_video)
    frames = torch.as_tensor(np.stack(frames))
    # 只对彩色图做图像增广（亮度、对比度、色调、饱和度等等），只做空间随机裁剪
    if DO_COLOR_AUGUMENT:
        frames = transform.color_jitter(
            frames,
            img_brightness=0.2,
            img_contrast=0,
            img_saturation=0,
            mode='L'
        )
    # print(type(frames), frames.shape)
    # tensor [100, 3, 320, 300]
    # 然后对当前文件夹下所有图像做时序采样(如果不够帧数，则循环复制）
    # frames = transform.TemporalRandomCrop(size=num_frames,
    #                                       downsample=sampling_rate)(frames)

    # 如果帧数不够，则复制
    if frames.shape[0] != num_frames:
        index = torch.linspace(0, frames.shape[0], num_frames)
        index = torch.clamp(index, 0, frames.shape[0] - 1).long()
        frames = torch.index_select(frames, 0, index)

    # 随机水平翻转(GOP为单位）
    if DO_COLOR_AUGUMENT:
        frames, _ = transform.horizontal_flip(0.5, frames)
    # print(2, len(frames), path_to_video)

    # print(type(frames), frames.shape)
    # tensor [32, 3, 320, 300]
    return frames


def smoke_decoder_gray_style_0821(
        path_to_video,
        sampling_rate,
        num_frames,
        clip_idx,
        num_clips,
        # min_scale,
        # max_scale,
        target_scale,
        phase,
        half_face=False,
):
    '''
    :param path_to_video:
    :param sampling_rate: self.cfg.DATA.SAMPLING_RATE, 每隔几帧取一帧。如果文件夹内图像很多的话，就间隔取；如果图像不够指定帧，则不间隔
    :param num_frames: self.cfg.DATA.NUM_FRAMES, Fast分支需要的帧数
    :param clip_idx: temporal_sample_index,
    :param num_clips: self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
    :param target_fps: self.cfg.DATA.TARGET_FPS,（图像数据集不需要这个参数）
    :param fps: 30，与target_fps一致（图像数据集不需要这个参数）
    :return:

    frames = decoder.wheel_decoder(
        self._path_to_videos[index],
        sampling_rate=self.cfg.DATA.SAMPLING_RATE,
        num_frames=self.cfg.DATA.NUM_FRAMES,
        clip_idx=temporal_sample_index,
        num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
        target_fps=self.cfg.DATA.TARGET_FPS,
        min_scale=min_scale,
        max_scale=max_scale
    )

    0821 updates
    添加人脸先验信息，读取mod txt的人脸bbox

    '''

    # 如果是测试阶段，或者是灰度图，则不需要进行图像增广（亮度、对比度、色调、饱和度等等），
    # 只做空间随机裁剪
    if phase != 'train':
        DO_COLOR_AUGUMENT = False
    else:
        DO_COLOR_AUGUMENT = True
        COLOR_AUGUMENT = transform.Compose([
            # transform.RandomResize(),
            transform.RandomRotate(),
            # transform.Gaussian_blur(),
            transform.SaltImage(),
        ])

    txt_path = "/data1/smoke/" + \
               "/".join(path_to_video.split("/")[5:]) + \
               "/dbd_mod_res_1010_0327_add_offset.txt"
    if not os.path.exists(txt_path):
        return None

    lines = open(txt_path, 'r').readlines()

    # 排除前后各15%
    thresh = int(0.15 * len(lines))
    lines = lines[thresh: -thresh]

    frames = []  # 保存待读取的帧
    # 读取每一行
    for idx, line in enumerate(lines):
        if len(frames) == num_frames:
            break

        # 等间隔采样
        if idx % sampling_rate:
            line = line.strip().split()
            # 当前帧包含人脸
            try:
                if line[1] != 0:
                    x1, x2, y1, y2 = map(int, line[9:13])  # 人脸坐标
                    offset_x = (x2 - x1) * 0.2  # 稍微往下取一点，包括烟的区域
                    offset_y = (y2 - y1) * 0.2

                    img_path = os.path.join(path_to_video + "/", line[0])
                    # print(txt_path)
                    # print(img_path)
                    img = Image.open(img_path).convert("L")
                    if phase in ['train', 'val']:
                        if not half_face:
                            img = img.resize((1280, 720)).crop((x1 - offset_x,
                                                                y1,
                                                                x2 + offset_x,
                                                                y2 + offset_y))
                        elif half_face:
                            assert 'not support'
                    elif phase == 'test':
                        if not half_face:
                            img = img.resize((1280, 720)).crop((x1 - offset_x,
                                                                y1,
                                                                x2 + offset_x,
                                                                y2 + offset_y))

                        elif half_face:
                            assert 'not support'

                    img = img.resize((target_scale, target_scale))  # 按要求缩放
                    if DO_COLOR_AUGUMENT:
                        img = COLOR_AUGUMENT(img)  # 随机、旋转、椒盐噪声等
                    # img_tensor = ToTensor()(img)  # raw version
                    img_tensor = ToTensor()(
                        ToNumpy()(img))  # 修改numpy array not writable错误
                    frames.append(img_tensor)
            except:
                continue
    if len(frames) == 0:
        return None

    # print(1, len(frames), path_to_video)
    frames = torch.as_tensor(np.stack(frames))
    # 只对彩色图做图像增广（亮度、对比度、色调、饱和度等等），只做空间随机裁剪
    if DO_COLOR_AUGUMENT:
        frames = transform.color_jitter(
            frames,
            img_brightness=0.2,
            img_contrast=0,
            img_saturation=0,
            mode='L'
        )
    # print(type(frames), frames.shape)
    # tensor [100, 3, 320, 300]
    # 然后对当前文件夹下所有图像做时序采样(如果不够帧数，则循环复制）
    # frames = transform.TemporalRandomCrop(size=num_frames,
    #                                       downsample=sampling_rate)(frames)

    # 如果帧数不够，则复制
    if frames.shape[0] != num_frames:
        index = torch.linspace(0, frames.shape[0], num_frames)
        index = torch.clamp(index, 0, frames.shape[0] - 1).long()
        frames = torch.index_select(frames, 0, index)

    # 随机水平翻转(GOP为单位）
    if DO_COLOR_AUGUMENT:
        frames, _ = transform.horizontal_flip(0.5, frames)
    # print(2, len(frames), path_to_video)

    # print(type(frames), frames.shape)
    # tensor [32, 3, 320, 300]
    return frames


def wheel_decoder_for_vis(
        path_to_video,
        sampling_rate,
        num_frames,
        clip_idx,
        num_clips,
        target_scale,
):
    '''
    用于可视化特征图
    :param path_to_video:
    :param sampling_rate: self.cfg.DATA.SAMPLING_RATE, 每隔几帧取一帧。如果文件夹内图像很多的话，就间隔取；如果图像不够指定帧，则不间隔
    :param num_frames: self.cfg.DATA.NUM_FRAMES, Fast分支需要的帧数
    :param clip_idx: temporal_sample_index,
    :param num_clips: self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
    :param target_fps: self.cfg.DATA.TARGET_FPS,（图像数据集不需要这个参数）
    :param fps: 30，与target_fps一致（图像数据集不需要这个参数）
    :return:

    frames = decoder.wheel_decoder(
        self._path_to_videos[index],
        sampling_rate=self.cfg.DATA.SAMPLING_RATE,
        num_frames=self.cfg.DATA.NUM_FRAMES,
        clip_idx=temporal_sample_index,
        num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
        target_fps=self.cfg.DATA.TARGET_FPS,
        min_scale=min_scale,
        max_scale=max_scale
    )

    '''
    # 输入文件夹路径，读取文件夹内的所有图片，并执行时间采样
    images = os.listdir(path_to_video)
    frame_count = len(images)  # 这段视频的总帧数

    # 每隔几帧取一帧。如果文件夹内图像很多的话，就间隔取；如果图像不够指定帧，则不间隔
    if frame_count < num_frames * sampling_rate:
        sampling_rate = 1

    # 先读取一帧，得到当前文件夹内图像的大致宽高信息
    img = Image.open(os.path.join(path_to_video, images[0]))
    frame_width = int(img.size[0])
    frame_height = int(img.size[1])

    DO_COLOR_AUGUMENT = False

    # 存储当前文件夹下包含的所有图像
    # 注意，ADAS数据集里，即便同一个文件夹内的图像大小也不一致，需要先做resize

    # 随机缩放：
    # 1. 从左上角区域随机选取起始点
    # 2. 从左上角开始，截取到右下角的全部区域
    # 3. 把截取到的图像resize成指定大小
    start_width = np.random.randint(0, 0.1 * frame_width)
    start_height = np.random.randint(0, 0.1 * frame_height)

    # 计算需要采样的帧的索引（只读取用得到的帧，可以节省大量的时间）
    start_idx, end_idx = get_start_end_idx_in_the_middle(
        frame_count,
        num_frames * sampling_rate,
        clip_idx,
        num_clips,
    )
    frames = []
    raw_imgs = []  # 用于可视化
    img_paths = []  # 路径
    for idx, img in enumerate(images):
        # 只读取需要的帧
        if idx < start_idx:
            continue
        if idx > end_idx:
            break
        if idx % sampling_rate == 0:
            img_paths.append(os.path.join(path_to_video, img))
            rawimg = Image.open(os.path.join(path_to_video, img))
            raw_imgs.append(rawimg)
            img = rawimg.crop(
                (start_width, start_height, frame_width, frame_height))
            img = img.resize((target_scale, target_scale))  # 按要求缩放
            # img_tensor = ToTensor()(img)  # raw version
            img_tensor = ToTensor()(
                ToNumpy()(img))  # 修改numpy array not writable错误
            frames.append(img_tensor)

    # print(1, len(frames), path_to_video)
    frames = torch.as_tensor(np.stack(frames))
    # print(frames.shape)

    # 如果帧数不够，则复制
    if frames.shape[0] != num_frames:
        index = torch.linspace(0, frames.shape[0], num_frames)
        index = torch.clamp(index, 0, frames.shape[0] - 1).long()
        frames = torch.index_select(frames, 0, index)

    # 随机水平翻转(GOP为单位）
    frames, _ = transform.horizontal_flip(0.5, frames)
    # print(2, len(frames), path_to_video)

    # print(type(frames), frames.shape)
    # tensor [32, 3, 320, 300]
    frames = frames.permute(0, 2, 3, 1)
    # raw kinetics video decoder
    # T H W C -> C T H W.
    # torch.Size([12, 256, 454, 3]) -> [3, 12, 256, 454]
    # frames = frames.permute(3, 0, 1, 2)
    # wdf wheel image decoder
    # T C H W -> C T H W
    # torch.Size([8, 3, 350, 324])
    # Perform data augmentation.
    # logger.info("frames shape after permute {}...".format(frames.shape))
    # print("frames shape after permute {}...".format(frames.shape))
    # Perform color normalization.
    # 注意，归一化这一步，即便输入的是灰度图（只有一个通道），经过减均值的broadcast机制，也变成了3通道
    frames = utils.tensor_normalize(
        frames, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )

    # 再调整回支持训练的格式[C T H W]
    frames = frames.permute(3, 0, 1, 2)
    return frames.unsqueeze(0), raw_imgs, img_paths


def wheel_decoder_for_vis_gray(
        path_to_video,
        sampling_rate,
        num_frames,
        clip_idx,
        num_clips,
        target_scale,
):
    '''
    用于可视化特征图 灰度版本
    :param path_to_video:
    :param sampling_rate: self.cfg.DATA.SAMPLING_RATE, 每隔几帧取一帧。如果文件夹内图像很多的话，就间隔取；如果图像不够指定帧，则不间隔
    :param num_frames: self.cfg.DATA.NUM_FRAMES, Fast分支需要的帧数
    :param clip_idx: temporal_sample_index,
    :param num_clips: self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
    :param target_fps: self.cfg.DATA.TARGET_FPS,（图像数据集不需要这个参数）
    :param fps: 30，与target_fps一致（图像数据集不需要这个参数）
    :return:

    frames = decoder.wheel_decoder(
        self._path_to_videos[index],
        sampling_rate=self.cfg.DATA.SAMPLING_RATE,
        num_frames=self.cfg.DATA.NUM_FRAMES,
        clip_idx=temporal_sample_index,
        num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
        target_fps=self.cfg.DATA.TARGET_FPS,
        min_scale=min_scale,
        max_scale=max_scale
    )

    '''
    # 输入文件夹路径，读取文件夹内的所有图片，并执行时间采样
    images = os.listdir(path_to_video)
    frame_count = len(images)  # 这段视频的总帧数

    # 每隔几帧取一帧。如果文件夹内图像很多的话，就间隔取；如果图像不够指定帧，则不间隔
    if frame_count < num_frames * sampling_rate:
        sampling_rate = 1

    # 先读取一帧，得到当前文件夹内图像的大致宽高信息
    img = Image.open(os.path.join(path_to_video, images[0]))
    frame_width = int(img.size[0])
    frame_height = int(img.size[1])

    DO_COLOR_AUGUMENT = False

    # 存储当前文件夹下包含的所有图像
    # 注意，ADAS数据集里，即便同一个文件夹内的图像大小也不一致，需要先做resize

    # 随机缩放：
    # 1. 从左上角区域随机选取起始点
    # 2. 从左上角开始，截取到右下角的全部区域
    # 3. 把截取到的图像resize成指定大小
    start_width = np.random.randint(0, 0.1 * frame_width)
    start_height = np.random.randint(0, 0.1 * frame_height)

    # 计算需要采样的帧的索引（只读取用得到的帧，可以节省大量的时间）
    start_idx, end_idx = get_start_end_idx_in_the_middle(
        frame_count,
        num_frames * sampling_rate,
        clip_idx,
        num_clips,
    )
    frames = []
    raw_imgs = []  # 用于可视化
    img_paths = []  # 路径
    for idx, img in enumerate(images):
        # 只读取需要的帧
        if idx < start_idx:
            continue
        if idx > end_idx:
            break
        if idx % sampling_rate == 0:
            img = Image.open(os.path.join(path_to_video, img)).convert("L")
            img = img.crop(
                (start_width, start_height, img.size[0], img.size[1]))
            img = img.resize((target_scale, target_scale))  # 按要求缩放
            # img_tensor = ToTensor()(img)  # raw version
            img_tensor = ToTensor()(
                ToNumpy()(img))  # 修改numpy array not writable错误
            frames.append(img_tensor)

    # print(1, len(frames), path_to_video)
    frames = torch.as_tensor(np.stack(frames))
    # print(frames.shape)

    # 如果帧数不够，则复制
    if frames.shape[0] != num_frames:
        index = torch.linspace(0, frames.shape[0], num_frames)
        index = torch.clamp(index, 0, frames.shape[0] - 1).long()
        frames = torch.index_select(frames, 0, index)

    # 随机水平翻转(GOP为单位）
    frames, _ = transform.horizontal_flip(0.5, frames)
    # print(2, len(frames), path_to_video)

    # print(type(frames), frames.shape)
    # tensor [32, 3, 320, 300]
    frames = frames.permute(0, 2, 3, 1)
    # raw kinetics video decoder
    # T H W C -> C T H W.
    # torch.Size([12, 256, 454, 3]) -> [3, 12, 256, 454]
    # frames = frames.permute(3, 0, 1, 2)
    # wdf wheel image decoder
    # T C H W -> C T H W
    # torch.Size([8, 3, 350, 324])
    # Perform data augmentation.
    # logger.info("frames shape after permute {}...".format(frames.shape))
    # print("frames shape after permute {}...".format(frames.shape))
    # Perform color normalization.
    # 注意，归一化这一步，即便输入的是灰度图（只有一个通道），经过减均值的broadcast机制，也变成了3通道
    frames = utils.tensor_normalize(
        frames, [0.48], [0.229]
    )

    # 再调整回支持训练的格式[C T H W]
    frames = frames.permute(3, 0, 1, 2)
    return frames.unsqueeze(0), raw_imgs, img_paths
