import copy
import numpy as np
import const
from pointLight import PointLight
from directionalLight import DirectionalLight


class Triangle:

    a = np.array([0, 0, -1, 1])
    b = np.array([1, 0, -1, 1])
    c = np.array([0, 1, -1, 1])
    n = np.array([0, 0, 1, 0])
    transformation = np.eye(4)
    material = {'ambient': np.array([.2, .2, .2]), 'diffuse': np.array([0, 0, 0]), 'specular': np.array([0, 0, 0]),
                'shininess': 0, 'emission': np.array([0, 0, 0]), }
    inv = np.eye(4)

    # counter-clockwise
    def __init__(self, a=np.array([0, 0, -1, 1]), b=np.array([1, 0, -1, 1]), c=np.array([0, 1, -1, 1])):
        self.a = a
        self.b = b
        self.c = c

        ab = (b - a)[0:3]
        ac = (c - a)[0:3]
        n = np.cross(ab, ac)
        n = n / np.linalg.norm(n)

        self.n = np.array([n[0], n[1], n[2], 0])

    def set_transformation(self, m):
        self.transformation = m
        self.inv = np.linalg.inv(m)

    def intersect(self, ray):
        intersection = None
        ray1 = ray.transform(self.inv)

        if np.abs(np.dot(ray1.direction, self.n)) < const.EPS:
            return intersection

        t = (np.dot(self.a, self.n) - np.dot(ray1.pos, self.n)) / np.dot(ray1.direction, self.n)

        if t <= 0:
            return intersection

        p = ray1.pos + ray1.direction * t

        if self._is_in(p):
            p = np.matmul(self.transformation, p)
            p = p / p[3]
            n = np.matmul((np.linalg.inv(self.transformation[0:3, 0:3])).T, self.n[0:3])
            n = n / np.linalg.norm(n)
            n = np.hstack([n, 0])

            intersection = {'primitive': self, 'point': p, 'normal': n}

        return intersection

    def _is_in(self, p):
        ab = self.b - self.a
        ac = self.c - self.a
        ap = p - self.a

        m = np.array([ab, ac])

        for i in [[0, 1], [0, 2], [1, 2]]:
            a = m[:, i]
            if abs(np.linalg.det(a)) > 0:
                beta, gamma = np.linalg.solve(a.T, ap[i])
                return 0 <= beta and 0 <= gamma and beta + gamma <= 1

    def aggregate_intersect(self, ray):
        intersection = None
        if np.abs(np.dot(ray.direction, self.n)) < const.EPS:
            return intersection, -1
        t = (np.dot(self.a, self.n) - np.dot(ray.pos, self.n)) / np.dot(ray.direction, self.n)
        if t <= 0:
            return intersection, t
        p = ray.pos + ray.direction * t
        if self._is_in(p):
            intersection = {'primitive': self, 'point': p, 'normal': self.n}
        return intersection, t

    def is_aggregate_intersect(self, ray, light):
        if np.abs(np.dot(ray.direction, self.n)) < const.EPS:
            return False
        t = (np.dot(self.a, self.n) - np.dot(ray.pos, self.n)) / np.dot(ray.direction, self.n)
        if t <= 0:
            return False
        p = ray.pos + ray.direction * t
        if self._is_in(p):
            if isinstance(light, DirectionalLight):
                return True
            elif isinstance(light, PointLight):
                light_pos_in_obj = np.matmul(self.inv, light.pos)
                light_pos_in_obj = light_pos_in_obj / light_pos_in_obj[3]
                d2 = np.linalg.norm(ray.pos - light_pos_in_obj)
                if t < d2:
                    return True
        return False

    def is_intersect(self, ray, light):
        ray1 = ray.transform(self.inv)
        ray1.pos = ray1.pos / ray1.pos[3]
        return self.is_aggregate_intersect(ray1, light)

    def __str__(self):
        s = 'triangle:'
        s += ' (%.2f, %.2f, %.2f)' % (self.a[0], self.a[1], self.a[2])
        s += ' (%.2f, %.2f, %.2f)' % (self.b[0], self.b[1], self.b[2])
        s += ' (%.2f, %.2f, %.2f)' % (self.c[0], self.c[1], self.c[2])
        return s
