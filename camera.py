import numpy as np
from ray import Ray


class Camera:

    def __init__(self, eye=np.array([0, 0, 0, 1]), look_at=np.array([0, 0, -1, 0]), up=np.array([0, 1, 0, 0]), fov=90):
        self.eye = eye
        self.look_at = look_at
        self.up = up
        self.fov = fov  # degrees

    def generate_ray(self, sample, width, height):
        ray = Ray()
        ray.pos = self.eye

        u, v, w = self._uvw()

        i = sample[0]
        j = sample[1]

        aspect = width / height
        a = np.tan(np.deg2rad(self.fov/2)) * aspect * (j - width / 2) / (width / 2)
        b = np.tan(np.deg2rad(self.fov/2)) * (height / 2 - i) / (height / 2)

        d = a * u + b * v - w
        ray.direction = np.hstack( [d / np.linalg.norm(d), 0] )

        return ray

    def _uvw(self):
        w = (self.eye - self.look_at)[0:3]
        w = w / np.linalg.norm(w)

        self.up = self._upvector(self.up[0:3], (self.eye - self.look_at)[0:3])
        u = np.cross(self.up, w)
        u = u / np.linalg.norm(u)

        v = np.cross(w, u)
        v = v / np.linalg.norm(v)

        return u, v, w

    def _upvector(self, up, z_vec):
        x = np.cross(up, z_vec)
        y = np.cross(z_vec, x)
        return y / np.linalg.norm(y)