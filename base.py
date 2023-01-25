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
    def qm(quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        
        return np.array([
            -x1*x0 - y1*y0 - z1*z0 + w1*w0,
            x1*w0 - y1*z0 + z1*y0 + w1*x0,
            x1*z0 + y1*w0 - z1*x0 + w1*y0,
            -x1*y0 + y1*x0 + z1*w0 + w1*z0
        ])

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

        self._inv = np.array([1, -1, -1, -1])

    def world_to_camera(self, world_coor: List[np.ndarray]) -> List[np.ndarray]:
        camera_coor = [self.qm(self.qm(self.quaternion, coor), self.quaternion*self._inv) for coor in world_coor]

        return world_coor
