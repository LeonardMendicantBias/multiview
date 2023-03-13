from dataclasses import dataclass, field
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import os
import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from datetime import datetime

from typing import List

import requests
from pysolotools.consumers import Solo
from pysolotools.converters.solo2coco import SOLO2COCOConverter
from pysolotools.core.models import KeypointAnnotationDefinition, RGBCameraCapture
from pysolotools.core.models import Frame
from typing import Dict, List, Tuple

import torch
# from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from torchvision.models import resnet18, ResNet18_Weights

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# config_file = '../../Projects/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
# checkpoint_file = '../../Projects/mmdetection/checkpoints/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth'

# # build the model from a config file and a checkpoint file
# obj_det = init_detector(config_file, checkpoint_file, device='cuda:0')

# np.set_printoptions(suppress=True)
# %matplotlib ipympl

class CustomConverter(SOLO2COCOConverter):

    categories = [
        {
            "supercategory": "default",
            "id": 1,
            "name": "person",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 2,
            "name": "bicycle",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 3,
            "name": "car",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 4,
            "name": "motorcycle",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 5,
            "name": "airplane",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 6,
            "name": "bus",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 7,
            "name": "train",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 8,
            "name": "truck",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 9,
            "name": "boat",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 10,
            "name": "traffic light",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 11,
            "name": "fire hydrant",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 12,
            "name": "stop sign",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 13,
            "name": "parking meter",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 14,
            "name": "bench",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 15,
            "name": "bird",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 16,
            "name": "cat",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 17,
            "name": "dog",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 18,
            "name": "horse",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 19,
            "name": "sheep",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 20,
            "name": "cow",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 21,
            "name": "elephant",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 22,
            "name": "bear",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 23,
            "name": "zebra",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 24,
            "name": "giraffe",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 25,
            "name": "backpack",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 26,
            "name": "umbrella",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 27,
            "name": "handbag",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 28,
            "name": "tie",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 29,
            "name": "suitcase",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 30,
            "name": "frisbee",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 31,
            "name": "skis",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 32,
            "name": "snowboard",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 33,
            "name": "sports ball",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 34,
            "name": "kite",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 35,
            "name": "baseball bat",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 36,
            "name": "baseball glove",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 37,
            "name": "skateboard",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 38,
            "name": "surfboard",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 39,
            "name": "tennis racket",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 40,
            "name": "bottle",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 41,
            "name": "wine glass",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 42,
            "name": "cup",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 43,
            "name": "fork",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 44,
            "name": "knife",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 45,
            "name": "spoon",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 46,
            "name": "bowl",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 47,
            "name": "banana",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 48,
            "name": "apple",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 49,
            "name": "sandwich",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 50,
            "name": "orange",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 51,
            "name": "broccoli",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 52,
            "name": "carrot",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 53,
            "name": "hot dog",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 54,
            "name": "pizza",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 55,
            "name": "donut",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 56,
            "name": "cake",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 57,
            "name": "chair",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 58,
            "name": "couch",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 59,
            "name": "potted plant",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 60,
            "name": "bed",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 61,
            "name": "dining table",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 62,
            "name": "toilet",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 63,
            "name": "tv",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 64,
            "name": "laptop",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 65,
            "name": "mouse",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 66,
            "name": "remote",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 67,
            "name": "keyboard",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 68,
            "name": "cell phone",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 69,
            "name": "microwave",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 70,
            "name": "oven",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 71,
            "name": "toaster",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 72,
            "name": "sink",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 73,
            "name": "refrigerator",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 74,
            "name": "book",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 75,
            "name": "clock",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 76,
            "name": "vase",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 77,
            "name": "scissors",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 78,
            "name": "teddy bear",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 79,
            "name": "hair drier",
            "keypoints": [],
            "skeleton": []
        },
        {
            "supercategory": "default",
            "id": 80,
            "name": "toothbrush",
            "keypoints": [],
            "skeleton": []
        }
    ]
    
    def _categories(self):
        return self.categories
    
    def callback(self, results):
        for result in results:
            super().callback(result)

    @staticmethod
    def _process_instances(
        frame: Frame, idx, output, data_root, solo_kp_map
    ) -> Tuple[Dict, List, List, List]:
        # logger.info(f"Processing Frame number: {idx}")
        image_id = idx
        sequence_num = frame.sequence
        rgb_captures = list(
            filter(lambda cap: isinstance(cap, RGBCameraCapture), frame.captures)
        )

        results = []
        for capture_idx, rgb_capture in enumerate(rgb_captures):
            img_record = SOLO2COCOConverter._process_rgb_image(
                image_id*len(rgb_captures)+capture_idx, rgb_capture, output, data_root, sequence_num
                # f'{image_id}_{capture_idx}', rgb_capture, output, data_root, sequence_num
            )
            (
                ann_record,
                ins_ann_record,
                sem_ann_record,
            ) = SOLO2COCOConverter._process_annotations(
                image_id*len(rgb_captures)+capture_idx, rgb_capture, sequence_num, data_root, solo_kp_map
                # f'{image_id}_{capture_idx}', rgb_capture, sequence_num, data_root, solo_kp_map
            )
            results.append(
                (img_record, ann_record, ins_ann_record, sem_ann_record)
            )
        return results
    

    def convert(self, output_path: str, dataset_name: str = "coco"):
        output = os.path.join(output_path, dataset_name)

        solo_kp_map = self._get_solo_kp_map()

        for idx, frame in enumerate(self._solo.frames()):
            self._pool.apply_async(
                self._process_instances,
                args=(frame, idx, output, self._solo.data_path, solo_kp_map),
                callback=self.callback,
            )
        # processed = map(self._process_instances, )

        self._pool.close()
        self._pool.join()

        self._write_out_annotations(output)

        # return img_record, ann_record, ins_ann_record, sem_ann_record

if __name__ == "__main__":
    folder_dir = 'C:/Users/Leonard/AppData/LocalLow/DefaultCompany/Perception2/solo_16'

    solo = Solo(data_path=folder_dir)
    dataset = CustomConverter(solo)
    dataset.convert(output_path='./data/')
