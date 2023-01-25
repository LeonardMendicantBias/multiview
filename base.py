import json
from dataclasses import dataclass, field
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R  # rotation axis ??? left-hand / clockwise


@dataclass
class Camera:
    position: np.ndarray=np.array([0., 0., 0.])
    quaternion: np.ndarray=np.array([1., 0., 0., 0.])  # angle in quaternion

    resolution: np.ndarray=np.array([3840, 2160])
    proj: np.ndarray=np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    sensor_size: np.ndarray=np.array([30, 30])

    @staticmethod
    def _convert_to_ndarray(obj):
        if not isinstance(obj, (np.ndarray, np.generic)):
            return np.array(obj)
        return obj

    def __post_init__(self):
        self.position = self._convert_to_ndarray(self.position)
        self.quaternion = self._convert_to_ndarray(self.quaternion)
        self.resolution = self._convert_to_ndarray(self.resolution)
        self.sensor_size = self._convert_to_ndarray(self.sensor_size)

    def world_to_camera(self, world_coor: List[np.ndarray]) -> List[np.ndarray]:
        q = R.from_quat(self.quaternion)
        camera_coor = q.apply(world_coor - self.position)

        return world_coor
