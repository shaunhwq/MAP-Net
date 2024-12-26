import os
from typing import Tuple

import cv2
import torch
import numpy as np

from mmedit.models import build_model


def pre_process(image: np.array, device: str, mean: Tuple[int] = (0.485, 0.456, 0.406), std: Tuple[int] = (0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    Pipeline from  configs/dehazers/_base_/datasets/hazeworld.py

    lq_pipeline = [
        dict(
            type='GenerateFileIndices',
            interval_list=[1],
            annotation_tree_json='data/HazeWorld/test/meta_info_tree_GT_test.json'
        ),
        dict(
            type='LoadImageFromFileList',
            io_backend=io_backend,
            key='lq',
            flag='unchanged',
            # channel_order='rgb'
        ),
        dict(
            type='RescaleToZeroOne',
            keys=['lq', 'gt']),
        dict(
            type='Normalize',
            keys=['lq'],
            **dict(
	            mean=[0.485, 0.456, 0.406],
	            std=[0.229, 0.224, 0.225],
                to_rgb=True,
            )
	    ),
        dict(
            type='FramesToTensor',
            keys=['lq', 'gt']
        ),
    ]

    :param image: Image produced by cv2, in the format BGR, with channels [h, w, c]
    :param device: device string to send the image tensor to.
    :returns: Processed tensor, ready to be stacked. [c, h, w] dims
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0

    # Normalization
    image_rgb = cv2.subtract(image_rgb, np.float64(np.array(mean).reshape(1, -1)))
    image_rgb = cv2.multiply(image_rgb, 1 / np.float64(np.array(std).reshape(1, -1)))

    image_np = np.ascontiguousarray(np.transpose(image_rgb, (2, 0, 1)))
    image_tensor = torch.from_numpy(image_np)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor


def post_process(model_output: torch.Tensor) -> np.array:
    """
    Pipeline from  configs/dehazers/_base_/datasets/hazeworld.py

    gt_pipeline = [
        dict(
            type='GenerateFileIndices',
            interval_list=[1],
            annotation_tree_json='data/HazeWorld/test/meta_info_tree_GT_test.json'
        ),
        dict(
            type='LoadImageFromFileList',
            io_backend=io_backend,
            key='gt',
            flag='unchanged',
            **load_kwargs
        ),
        dict(
            type='RescaleToZeroOne',
            keys=['lq', 'gt']
        ),
        dict(
            type='Normalize',
            keys=['gt'],
            **dict(
                mean=[0., 0., 0.],
                std=[1., 1., 1.],
                to_rgb=True,
            )
        ),
        dict(
            type='FramesToTensor',
            keys=['lq', 'gt']
        ),
    ]

    Post-process does the inverse of the pipeline since we want to obtain numpy bgr image from torch

    :param model_output: Output produced by the model, [b, c, h, w], in RGB.
    :returns: Image used by cv2, in the format BGR, with channels [h, w, c]
    """
    image_rgb = model_output.cpu().squeeze().permute(1, 2, 0).numpy()
    image_rgb = (image_rgb * 255).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


if __name__ == "__main__":
    video_path = "/Users/shaun/datasets/image_enhancement/dehaze/DVD/DrivingHazy/31_hazy_video.mp4"
    weights_path = "mapnet_hazeworld_40k.pth"
    device = "mps"
    num_input_frames = 11
    downsize_scale = 1  # Downsize for inference

    if device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # model settings from configs/dehazers/mapnet/mapnet_hazeworld.py
    checkpoint = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa
    model_config = dict(
        type='MAP',
        generator=dict(
            type='MAPNetInference',
            backbone=dict(
                type='ConvNeXt',
                arch='tiny',
                out_indices=[0, 1, 2, 3],
                drop_path_rate=0.0,
                layer_scale_init_value=1.0,
                gap_before_final_norm=False,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint=checkpoint,
                    prefix='backbone.'
                ),
            ),
            neck=dict(
                type='ProjectionHead',
                in_channels=[96, 192, 384, 768],
                out_channels=64,
                num_outs=4
            ),
            upsampler=dict(
                type='MAPUpsampler',
                embed_dim=32,
                num_feat=32,
            ),
            channels=32,
            num_trans_bins=32,
            align_depths=(1, 1, 1, 1),
            num_kv_frames=[1, 2, 3],
        ),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    )

    # Remove wrapper code around the torch.nn core code
    # generator found at models/backbones/map_backbones/mapnet_net.py
    # MAPNetInference removes all self.training code and makes it infer for live videos
    model_wrapper = build_model(model_config, train_cfg=None, test_cfg=None)
    model_wrapper.load_state_dict(torch.load(weights_path, map_location="cpu")["state_dict"])
    model = model_wrapper.generator
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    while True:
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()

        for i in range(downsize_scale):
            frame = cv2.pyrDown(frame)

        pre_processed_frame = pre_process(frame, device)
        with torch.no_grad():
            model_output = model(pre_processed_frame)
        out_image = post_process(model_output)

        display_frame = np.vstack([frame, out_image])
        for i in range(downsize_scale):
            display_frame = cv2.pyrUp(display_frame)

        cv2.imshow("output", display_frame)
        key = cv2.waitKey(1)
        if key & 255 == 27:
            break

    model.reset_memory()
    cap.release()
    cv2.destroyAllWindows()
