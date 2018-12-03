import numpy as np
from pointLight import PointLight
from directionalLight import DirectionalLight


class Sphere:

    transformation = np.eye(4)
    material = {'ambient': np.array([.2, .2, .2]), 'diffuse': np.array([0, 0, 0]), 'specular': np.array([0, 0, 0]),
                'shininess': 1, 'emission': np.array([0, 0, 0]), }
    inv = np.eye(4)

    def __init__(self, x=0, y=0, z=-5, radius=2):
        self.center = np.array([x, y, z, 1])
        self.radius = radius

    def set_transformation(self, m):
        self.transformation = m
        self.inv = np.linalg.inv(m)

    def intersect(self, ray):
        ray1 = ray.transform(self.inv)

        det, a, b, c = self._solve(ray1)

        if det < 0:
            return None

        t1 = (-b-np.sqrt(det)) / 2 / a
        t2 = (-b+np.sqrt(det)) / 2 / a

        if t1 * t2 < 0:
            return None

        t = min(t1, t2)

        p = ray1.pos + ray1.direction * t
        n = p - self.center

        p = np.matmul(self.transformation, p)
        p = p / p[3]
        n = np.matmul((np.linalg.inv(self.transformation[0:3, 0:3])).T, n[0:3])
        n = n / np.linalg.norm(n)
        n = np.hstack([n, 0])

        intersection = {'primitive': self, 'point': p, 'normal': n}

        return intersection

    def aggregate_intersect(self, ray):
        intersection = None
        det, a, b, c = self._solve(ray)
        if det < 0:
            return intersection, -1
        t1 = (-b-np.sqrt(det)) / 2 / a
        t2 = (-b+np.sqrt(det)) / 2 / a
        if t1 * t2 <= 0:
            return intersection, -1
        t = min(t1, t2)
        p = ray.pos + ray.direction * t
        n = p - self.center
        intersection = {'primitive': self, 'point': p, 'normal': n}
        return intersection, t

    def is_aggregate_intersect(self, ray, light):
        det, a, b, c = self._solve(ray)
        if det < 0:
            return False
        t1 = (-b - np.sqrt(det)) / 2 / a
        t2 = (-b + np.sqrt(det)) / 2 / a
        if t1 * t2 <= 0:
            return False
        # t = min(t1, t2)
        # p = ray.pos + ray.direction * t

        if isinstance(light, DirectionalLight):
            return True
        elif isinstance(light, PointLight):
            t = min(t1, t2)
            light_pos_in_obj = np.matmul(self.inv, light.pos)
            light_pos_in_obj = light_pos_in_obj / light_pos_in_obj[3]
            # d1 = np.linalg.norm(ray.pos - p)
            d2 = np.linalg.norm(ray.pos - light_pos_in_obj)
            if t < d2:
                return True
        return False

    def is_intersect(self, ray, light):
        ray1 = ray.transform(self.inv)

        det, a, b, c = self._solve(ray1)

        if det < 0:
            return False

        t1 = (-b - np.sqrt(det)) / 2 / a
        t2 = (-b + np.sqrt(det)) / 2 / a

        if t1 * t2 <= 0:
            return False

        t = min(t1, t2)
        p = ray1.pos + ray1.direction * t
        p = np.matmul(self.transformation, p)
        p = p / p[3]

        if np.dot(p - ray.pos, ray.direction) <= 0:
            return False
        if isinstance(light, DirectionalLight):
            return True
        if isinstance(light, PointLight):
            d1 = np.linalg.norm(ray.pos - p)
            d2 = np.linalg.norm(ray.pos - light.pos)
            if d1 < d2:
                return True
        return False

    def _solve(self, ray):
        p0 = ray.pos[0:3]
        p1 = ray.direction[0:3]
        c = self.center[0:3]
        r = self.radius

        a = np.dot(p1, p1)
        b = 2 * np.dot(p1, p0 - c)
        c = np.dot(p0 - c, p0 - c) - r * r

        det = b * b - 4 * a * c
        return [det, a, b, c]
