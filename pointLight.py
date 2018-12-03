import numpy as np
from ray import Ray
import const


class PointLight:

    def __init__(self, pos=np.array([1, 1, 1, 1]), color=np.array([1, 0, 0])):
        self.pos = pos
        self.color = color
        self.attenuation = np.array([1, 0, 0])

    def get_ray(self, point):
        direction = self.pos - point
        direction = direction / np.linalg.norm(direction)
        ray = Ray(point + direction * const.OFFSET, direction)
        return ray

    def get_attenuation(self, point):
        r = np.linalg.norm(self.pos - point)
        a = self.attenuation[0] + self.attenuation[1] * r + self.attenuation[2] * r * r
        return a
