import copy

import numpy as np

import transform
from directionalLight import DirectionalLight
from pointLight import PointLight
from scene import Scene
from sphere import Sphere
from triangle import Triangle
from tileSampler import TileSampler


class Parser:
    def __init__(self):
        # General
        self.scene = Scene()
        # Geometry
        self._vertex = []

        # Transformations
        self._transform_stack = []

        # Light
        self._attenuation = np.array([1, 0, 0])

        # Materials
        self._material = {'ambient': np.array([.2, .2, .2]), 'diffuse': np.array([0, 0, 0]),
                          'specular': np.array([0, 0, 0]), 'shininess': 0, 'emission': np.copy([0, 0, 0])}

        # self.aggregate = [AggregatePrimitive()]

    def _push(self):
        if not self._transform_stack:
            self._transform_stack.append(np.eye(4))
        self._transform_stack.append(copy.copy(self._transform_stack[-1]))

    def _pop(self):
        return self._transform_stack.pop()

    def _top(self):
        if not self._transform_stack:
            return np.eye(4)
        return self._transform_stack[-1]

    def parse(self, filename):
        with open(filename) as f:
            for s in f.readlines():
                if not s or s == '' or s[0] == '\n' or s[0] == '#':
                    continue

                s = s.split()

                if s[0] == 'size':
                    width, height = map(int, s[1:])
                    self.scene.film.init_image(width, height)
                    self.scene.sampler.tile_width = width
                    self.scene.sampler.tile_height = height
                elif s[0] == 'maxdepth':
                    max_depth = int(s[1])
                    self.scene.ray_tracer.max_depth = max_depth
                elif s[0] == 'output':
                    out_filename = s[1]
                    self.scene.film.filename = out_filename
                elif s[0] == 'camera':
                    eye_x, eye_y, eye_z, look_at_x, look_at_y, look_at_z, upx, upy, upz, fov = map(float, s[1:])
                    self.scene.camera.eye = np.array([eye_x, eye_y, eye_z, 1])
                    self.scene.camera.look_at = np.array([look_at_x, look_at_y, look_at_z, 1])
                    self.scene.camera.up = np.array([upx, upy, upz, 0])
                    self.scene.camera.fov = fov
                elif s[0] == 'sphere':
                    x, y, z, radius = map(float, s[1:])
                    sphere = Sphere(x, y, z, radius)
                    sphere.set_transformation(copy.copy(self._top()))
                    sphere.material = copy.copy(self._material)
                    self.scene.ray_tracer.primitives.append(sphere)
                elif s[0] == 'vertex':
                    x, y, z = map(float, s[1:])
                    self._vertex.append(np.array([x, y, z, 1]))
                elif s[0] == 'tri':
                    v1, v2, v3 = map(int, s[1:])
                    tri = Triangle(self._vertex[v1], self._vertex[v2], self._vertex[v3])
                    tri.set_transformation(copy.copy(self._top()))
                    tri.material = copy.copy(self._material)
                    self.scene.ray_tracer.primitives.append(tri)
                elif s[0] == 'translate':
                    x, y, z = map(float, s[1:])
                    self._transform_stack[-1] = np.matmul(self._transform_stack[-1], transform.translate(x, y, z))
                elif s[0] == 'rotate':
                    x, y, z, angle = map(float, s[1:])
                    a = np.array([x, y, z])
                    r = np.eye(4)
                    r[0:3, 0:3] = transform.rotate(angle, a)
                    self._transform_stack[-1] = np.matmul(self._transform_stack[-1], r)
                elif s[0] == 'scale':
                    x, y, z = map(float, s[1:])
                    self._transform_stack[-1] = np.matmul(self._transform_stack[-1], transform.scale(x, y, z))
                elif s[0] == 'pushTransform':
                    self._push()
                elif s[0] == 'popTransform':
                    self._pop()
                elif s[0] == 'directional':
                    x, y, z, r, g, b = map(float, s[1:])
                    l = DirectionalLight(np.array([x, y, z, 0]), np.array([r, g, b]))
                    self.scene.ray_tracer.lights.append(l)
                elif s[0] == 'point':
                    x, y, z, r, g, b = map(float, s[1:])
                    l = PointLight(np.array([x, y, z, 1]), np.array([r, g, b]))
                    l.attenuation = copy.copy(self._attenuation)
                    self.scene.ray_tracer.lights.append(l)
                elif s[0] == 'attenuation':
                    const, linear, quadratic = map(float, s[1:])
                    self._attenuation = np.array([const, linear, quadratic])
                elif s[0] == 'ambient':
                    r, g, b = map(float, s[1:])
                    self._material['ambient'] = np.array([r, g, b])
                elif s[0] == 'diffuse':
                    r, g, b = map(float, s[1:])
                    self._material['diffuse'] = np.array([r, g, b])
                elif s[0] == 'specular':
                    r, g, b = map(float, s[1:])
                    self._material['specular'] = np.array([r, g, b])
                elif s[0] == 'shininess':
                    shininess = float(s[1])
                    self._material['shininess'] = shininess
                elif s[0] == 'emission':
                    r, g, b = map(float, s[1:])
                    self._material['emission'] = np.array([r, g, b])

        return self.scene
