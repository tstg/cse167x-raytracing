import numpy as np
from ray import Ray
import const


class DirectionalLight:

    color = np.array([1, 1, 1])
    direction = np.array([0, 0, -1, 0])

    def __init__(self, direction, color):
        direction = direction / np.linalg.norm(direction)
        self.direction = direction
        self.color = color

    def get_ray(self, point):
        ray = Ray(point + self.direction * const.OFFSET, self.direction)
        return ray

    @staticmethod
    def get_attenuation(point):
        return 1
