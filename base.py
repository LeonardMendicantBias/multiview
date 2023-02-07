from dataclasses import dataclass, field
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R  # rotation axis ??? left-hand / clockwise


@dataclass
class Line:
    origin: np.ndarray
    direction: np.ndarray

    @classmethod
    def from_camera_coor(cls, origin, coor):
        return cls(np.array(origin), np.array(coor))
    
    def point(self, r):
        return self.origin + r*self.direction


def find_points(line_a: Line, line_b: Line):
    d = line_b.origin - line_a.origin
    u = np.dot(line_a.direction, line_b.direction)

    r_1 = ((np.dot(line_a.direction, d)) - u*np.dot(line_b.direction, d)) / (1 - u**2)
    r_2 = ((np.dot(line_b.direction, d)) - u*np.dot(line_a.direction, d)) / (u**2 - 1)

    p1 = line_a.point(r_1)
    p2 = line_b.point(r_2)

    return (p1 + p2) / 2
    
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
    def qm_2(quaternion, vector):
        q0, q1, q2, q3 = quaternion
        matrix = np.array([
            [1-2*q2**2-2*q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
            [2*(q1*q2-q0*q3), 1-2*q1**2-2*q3**2, 2*(q2*q3+q0*q1)],
            [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), 1-2*q1**2-2*q2**2]
        ])

        return np.matmul(vector, matrix.T)

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

        focal = 20.78461
        self.intrinsic = np.array([
            [focal*self.resolution[0]/self.sensor_size[0], 0, 0],
            [0, -focal*self.resolution[0]/self.sensor_size[0], 0],
            [self.resolution[0]/2, self.resolution[1]/2, 1],
        ])

    def world_to_camera(self, world_coor: List[np.ndarray]):
        world_coor = [coor - self.position for coor in world_coor]
        # world_coor = [np.insert(coor, 0, 0) for coor in world_coor]
        # camera_coor = [
        #     self.qm(self.qm(self.quaternion, coor), self.quaternion*self._inv)
        #     for coor in world_coor
        # ]
        camera_coor = [
            self.qm_2(self.quaternion, coor)
            for coor in world_coor
        ]
        return camera_coor

    def camera_to_pixel(self, camera_coor: List[np.ndarray]) -> List[np.ndarray]:
        ndc_coor = [np.matmul(coor, self.intrinsic) for coor in camera_coor]
        
        pixel_coor = [np.floor((coor/coor[-1])[:-1]) for coor in ndc_coor]
        # pixel_coor = [np.floor((coor/coor[-1])[:-1] + .5) for coor in ndc_coor]

        return pixel_coor

    def pixel_to_ray(self, pixel_coor):
        camera_coor = [
            np.matmul(np.insert(coor, 2, 1), np.linalg.inv(self.intrinsic))
            for coor in pixel_coor
        ]
        q_ = self.quaternion
        camera_coor = [
            self.qm_2(np.array([q_[0], -q_[1], -q_[2], -q_[3]]), coor)
            for coor in camera_coor
        ]

        return [Line.from_camera_coor(self.position, coor) for coor in camera_coor]


