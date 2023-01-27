from dataclasses import dataclass, field
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R  # rotation axis ??? left-hand / clockwise


@dataclass
class Line:
    origin: np.ndarray
    direction: np.ndarray


def find_points(line_a: Line, line_b: Line):
    n = np.cross(line_a.direction, line_b.direction)
    d = np.abs(np.dot(n, line_a.origin - line_b.origin)) / np.linalg.norm(n)
    
    t_a = np.dot(np.cross(line_b.direction, n), (line_b.origin - line_a.origin)) / np.dot(n, n)
    t_b = np.dot(np.cross(line_a.direction, n), (line_b.origin - line_a.origin)) / np.dot(n, n)

    p_a = line_a.origin + t_a * line_a.direction
    p_b = line_b.origin + t_b * line_b.direction

    return (p_a + p_b) / 2
    
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
            w1*w0 - x1*x0 - y1*y0 - z1*z0,
            w1*x0 + x1*w0 - y1*z0 + z1*y0,
            w1*y0 + x1*z0 + y1*w0 - z1*x0,
            w1*z0 - x1*y0 + y1*x0 + z1*w0
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
        self.proj = self._convert_to_ndarray(self.proj).reshape(3, 3)
        self.sensor_size = self._convert_to_ndarray(self.sensor_size)

        self._inv = np.array([1, -1, -1, -1])

        self.intrinsic = np.array([
            [0, 0, self.resolution[0]/2],
            [0, 0, self.resolution[1]/2],
            [0, 0, 0],
        ]) + self.proj

    # def world_to_camera(self, world_coor: List[np.ndarray]) -> List[np.ndarray]:
    #     world_coor = [np.insert(coor, 0, 0) for coor in world_coor]
    #     camera_coor = [self.qm(self.qm(self.quaternion, coor), self.quaternion*self._inv) for coor in world_coor]
    #     camera_coor = [coor[1:] - self.position for coor in camera_coor]
    #     # camera_coor = [np.matmul(c71.63506oor, self.proj) for coor in camera_coor]
    #     camera_coor = [
    #         (coor / (-coor[-1]))[:-1]
    #         for coor in camera_coor
    #     ]

    #     norm_coor = [(coor + self.sensor_size/2)/self.sensor_size for coor in camera_coor]
        
    #     raster_coor = [
    #         np.floor(np.array([
    #             coor[0]*self.resolution[0],
    #             coor[1]*self.resolution[1],
    #             # (1-coor[1])*self.resolution[1],
    #         ]))
    #         for coor in norm_coor
    #     ]

    #     return raster_coor

    def world_to_camera(self, world_coor: List[np.ndarray]) -> List[np.ndarray]:
        # world_coor = [np.insert(coor, 3, 0) for coor in world_coor]

        camera_coor = world_coor
        homo_camera_coor = [np.append(coor, 1) for coor in camera_coor]

        return homo_camera_coor
