import copy
import numpy as np
from ray import Ray
from directionalLight import DirectionalLight
from pointLight import PointLight
import const


class RayTracer:

    def __init__(self):
        self.max_depth = 2
        self.primitives = []
        self.lights = []

    def trace(self, ray, depth):
        if depth > self.max_depth:
            return const.BLACK
        intersection = self.intersect(ray)
        if not intersection:
            return const.BLACK
        color = self.shade(intersection, ray)
        # reflection. all of the vectors should be unit.
        obj_dir = ray.direction - 2 * np.dot(ray.direction, intersection['normal']) * intersection['normal']
        obj_dir = obj_dir / np.linalg.norm(obj_dir)
        obj_pos = intersection['point'] + obj_dir * const.OFFSET
        obj_ray = Ray(obj_pos, obj_dir)

        reflected_color = self.trace(obj_ray, depth + 1)

        color += intersection['primitive'].material['specular'] * reflected_color
        return color

    def intersect(self, ray):
        min_dist = np.inf
        intersection = None

        for primitive in self.primitives:
            tmp_intersection = primitive.intersect(ray)
            if tmp_intersection:
                dist = np.linalg.norm(tmp_intersection['point'] - ray.pos)
                if np.dot(tmp_intersection['point'] - ray.pos, ray.direction) > 0 and 0 < dist < min_dist:
                    min_dist = dist
                    intersection = tmp_intersection

        return intersection

    def shade(self, intersection, ray):
        color = intersection['primitive'].material['ambient'] + intersection['primitive'].material['emission']
        for light in self.lights:
            ray2light = light.get_ray(intersection['point'])
            intersection1 = self.intersect(ray2light)

            if isinstance(light, DirectionalLight) and intersection1:
                continue

            if isinstance(light, PointLight) and intersection1:
                d1 = np.linalg.norm(intersection['point'] - intersection1['point'])
                d2 = np.linalg.norm(intersection['point'] - light.pos)
                if d1 < d2:
                    continue

            if self._judge_light(ray2light, intersection, intersection1):
                half_vec = -ray.direction + ray2light.direction
                half_vec = half_vec / np.linalg.norm(half_vec)
                material = intersection['primitive'].material
                s = material['shininess']

                c = material['diffuse'] * max(np.dot(ray2light.direction, intersection['normal']), 0.0)
                c += material['specular'] * pow(max(np.dot(half_vec, intersection['normal']), 0.0), s)
                c *= light.color / light.get_attenuation(intersection['point'])
                color += c

        return color

    @staticmethod
    def _judge_light(light, intersection, intersection1):
        if isinstance(light, DirectionalLight):
            return not intersection1
        # point light
        if intersection1:
            d1 = np.linalg.norm(intersection['point'] - intersection1['point'])
            d2 = np.linalg.norm(intersection['point'] - light.pos)
            return d1 >= d2
        return not intersection1
