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
from typing import Dict, List, Tuple
from pysolotools.core.models import (
    Frame,
    Annotation, AnnotationLabel,
    BoundingBox2DAnnotation, BoundingBox2DLabel,
    BoundingBox3DAnnotation, BoundingBox3DLabel,
)

import torch
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from torchvision.models import resnet18, ResNet18_Weights

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from scipy.spatial.transform import Rotation as R


@dataclass
class Line:
    origin: np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        self.origin = np.array(self.origin)
        norm = np.linalg.norm(self.direction)
        if norm != 0: 
            self.direction = self.direction/norm
    
    def point(self, r):
        return self.origin + r*self.direction


@dataclass
class CustomBoundingBox2DLabel(BoundingBox2DLabel):
    data_path: str
    capture: RGBCameraCapture
    obj_det: torch.nn.Module
    resnet: torch.nn.Module
    preprocess: torch.nn.Module
    line: Line=field(init=False)
    # image: Image=field(init=False)
    # feature: np.ndarray=field(init=False)
    image: Image=None
    feature: np.ndarray=None

    @staticmethod
    def qm_2(quaternion, vector):
        q0, q1, q2, q3 = quaternion
        matrix = np.array([
            [1-2*q2**2-2*q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
            [2*(q1*q2-q0*q3), 1-2*q1**2-2*q3**2, 2*(q2*q3+q0*q1)],
            [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), 1-2*q1**2-2*q2**2]
        ])

        return np.matmul(vector, matrix.T)
    
    def __post_init__(self):
        # super().__post_init__()  # apparently the parent class does not have __post_init__() function

        # obtain the pixel coordinate of the box's center
        size = np.array(self.dimension)/2
        center = self.origin + size

        q_ = self.capture.rotation
        quaternion = np.array([q_[3], q_[0], q_[1], q_[2]])

        focal = 20.78461  # measured in millimeters
        dimension = self.capture.dimension
        sensor_size = np.array([30, 30])
        intrinsic = np.array([
            [focal*dimension[0]/sensor_size[0], 0, 0],
            [0, -focal*dimension[0]/sensor_size[1], 0],
            [dimension[0]/2, dimension[1]/2, 1],
        ])
        ndc_coor = np.matmul(np.array([*center, 1]), np.linalg.inv(intrinsic))
        camera_coor = self.qm_2(quaternion*np.array([1, -1, -1, -1]), ndc_coor)
        self.line = Line(self.capture.position, camera_coor)

        # # crop the image and extract the box's features
        # tl, br = center - size, center + size
        # image = Image.open(self.data_path)
        # self.image = image.crop((*tl, *br))

    @classmethod
    def from_parent(cls, bbox: BoundingBox2DLabel, capture: RGBCameraCapture, data_path, obj_det, resnet, preprocess):
        return cls(
            instanceId=bbox.instanceId,
            labelId=bbox.labelId,
            labelName=bbox.labelName,
            origin=bbox.origin,
            dimension=bbox.dimension,
            data_path=data_path,
            capture=capture,
            obj_det=obj_det,
            resnet=resnet,
            preprocess=preprocess,
        )


@dataclass
class CustomBoundingBox2DAnnotation(BoundingBox2DAnnotation):

    @classmethod
    def from_parent(cls, anno: BoundingBox2DAnnotation, capture: RGBCameraCapture, data_path, obj_det, resnet, preprocess):
        return cls(
            type=anno.type,
            id=anno.id,
            sensorId=anno.sensorId,
            description=anno.description,
            extra_data=anno.extra_data,
            values=[
                CustomBoundingBox2DLabel.from_parent(value, capture, data_path, obj_det, resnet, preprocess)
                for value in anno.values
            ],
        )


@dataclass
class CustomBoundingBox3DLabel(BoundingBox3DLabel):
    capture: RGBCameraCapture
    
    def __post_init__(self):
        r = R.from_quat(self.capture.rotation)
        self.loc = r.apply(self.translation) + self.capture.position

    @classmethod
    def from_parent(cls, bbox: BoundingBox3DLabel, capture: RGBCameraCapture):
        return cls(
            instanceId=bbox.instanceId,
            labelId=bbox.labelId,
            labelName=bbox.labelName,
            size=bbox.size,
            translation=bbox.translation,
            rotation=bbox.rotation,
            velocity=bbox.velocity,
            acceleration=bbox.acceleration,
            capture=capture
        )

@dataclass
class CustomBoundingBox3DAnnotation(BoundingBox3DAnnotation):

    @classmethod
    def from_parent(cls, anno: BoundingBox2DAnnotation, capture: RGBCameraCapture):
        return cls(
            type=anno.type,
            id=anno.id,
            sensorId=anno.sensorId,
            description=anno.description,
            extra_data=anno.extra_data,
            values=[
                CustomBoundingBox3DLabel.from_parent(value, capture)
                for value in anno.values
            ]
        )
    
@dataclass
class LocationLabel(AnnotationLabel):
    position: List[float]

    @classmethod
    def from_3DBBLabel(cls, bbox: BoundingBox3DLabel, capture: RGBCameraCapture):
        r = R.from_quat(capture.rotation)
        position = r.apply(bbox.translation) + capture.position
        return cls(
            instanceId=bbox.instanceId,
            labelId=bbox.labelId,
            position=position,
        )

@dataclass
class LocationAnnotation(Annotation):
    values: List[LocationLabel]

    @classmethod
    def from_3DAnnotation(cls, anno: BoundingBox3DAnnotation, capture: RGBCameraCapture):
        return cls(
            type=anno.type,
            id=anno.id,
            sensorId=anno.sensorId,
            description=anno.description,
            extra_data=anno.extra_data,
            values=[LocationLabel.from_3DBBLabel(value, capture) for value in anno.values]
        )
        
@dataclass
class FeatureLabel(AnnotationLabel):
    feature: np.ndarray

# @DataFactory.register("type.unity.com/unity.solo.DepthAnnotation")
@dataclass
class FeatureAnnotation(Annotation):
    values: List[LocationLabel]

    @classmethod
    def from_3DAnnotation(cls, anno: BoundingBox3DAnnotation, capture: RGBCameraCapture):
        return cls(
            type=anno.type,
            id=anno.id,
            sensorId=anno.sensorId,
            description=anno.description,
            extra_data=anno.extra_data,
            values=[LocationLabel.from_3DBBLabel(value, capture) for value in anno.values]
        )
    
@dataclass
class AugmentedRGBCameraCapture(RGBCameraCapture):

    def __post_init__(self):
        super().__post_init__()
        # convert the 'projection matrix' into a matrix for later use
        self._matrix = self.matrix  # save the original format
        self.matrix = np.array(self.matrix).reshape((3, 3))

        print(self.filename)

    @classmethod
    def from_rgb_capture(cls, data_path, rgb_capture: RGBCameraCapture, obj_det, resnet, preprocess):
        # data_path = f'{data_path}/{rgb_capture.filename}'
        # anno_3d = [anno for anno in rgb_capture.annotations if isinstance(anno, BoundingBox3DAnnotation)][0]
        # location_annotation = LocationAnnotation.from_3DAnnotation(anno_3d, rgb_capture)
        # for loc in location_annotation.values:
        #     print(loc)
        return cls(
            id=rgb_capture.id,
            type=rgb_capture.type,
            description=rgb_capture.description,
            position=rgb_capture.position,
            rotation=rgb_capture.rotation,
            # annotations=rgb_capture.annotations,
            annotations=[
                CustomBoundingBox2DAnnotation.from_parent(anno, rgb_capture, data_path, obj_det, resnet, preprocess)
                if isinstance(anno, BoundingBox2DAnnotation) else 
                CustomBoundingBox3DAnnotation.from_parent(anno, rgb_capture)
                for anno in rgb_capture.annotations
            ],
            velocity=rgb_capture.velocity,
            acceleration=rgb_capture.acceleration,
            filename=rgb_capture.filename,
            imageFormat=rgb_capture.imageFormat,
            dimension=rgb_capture.dimension,
            projection=rgb_capture.projection,
            matrix=rgb_capture.matrix,
        )


@dataclass
class DeepFrame(Frame):
    
    @staticmethod
    def _find_points(line_a: Line, line_b: Line):
        d = line_b.origin - line_a.origin
        u = np.dot(line_a.direction, line_b.direction)
        e = np.dot(line_a.direction, d)
        f = np.dot(line_b.direction, d)

        r_1 = (e - u*f) / (1 - u**2)
        r_2 = (f - u*e) / (u**2 - 1)

        p1 = line_a.point(r_1)
        p2 = line_b.point(r_2)

        return (p1 + p2) / 2
    
    def __post_init__(self):
        super().__post_init__()
        
        self.loc_gt = {}
        for captire_idx, capture in enumerate(self.captures):
            for anno_idx, anno in enumerate(capture.annotations):
                if isinstance(anno, CustomBoundingBox2DAnnotation):
                    for bbox_idx, bbox in enumerate(anno.values):
                        if bbox.instanceId not in self.loc_gt:
                            self.loc_gt[bbox.instanceId] = {'lines': []}
                        self.loc_gt[bbox.instanceId]['lines'].append(bbox.line)
                elif isinstance(anno, CustomBoundingBox3DAnnotation):
                    for bbox_idx, bbox in enumerate(anno.values):
                        if bbox.instanceId not in self.loc_gt:
                            self.loc_gt[bbox.instanceId] = {'lines': []}
                        self.loc_gt[bbox.instanceId]['loc'] = bbox.loc

        for key, item in self.loc_gt.items():
            self.loc_gt[key]['pred'] = self._find_points(*item['lines'])
            # print(loc_gt[key]['loc'])

        for instanceId, pred in self.loc_gt.items():
            loc = pred['loc']
            pred = pred['pred']
            self.loc_gt[instanceId]['pred_err'] = np.sqrt((pred-loc)**2).sum()

    @classmethod
    def from_frame(cls, frame: Frame, data_path: int, obj_det, resnet, preprocess):
        
        anno_3d = [anno for anno in frame.captures[0].annotations if isinstance(anno, BoundingBox3DAnnotation)][0]
        location_annotation = LocationAnnotation.from_3DAnnotation(anno_3d, frame.captures[0])
        # for loc in location_annotation.values:
        #     print(loc)
        return cls(
            frame=frame.frame,
            sequence=frame.sequence,
            step=frame.step,
            timestamp=frame.timestamp,
            metrics=frame.metrics,
            captures=[
                AugmentedRGBCameraCapture.from_rgb_capture(data_path, capture, obj_det, resnet, preprocess)
                for capture in frame.captures
            ],
        )