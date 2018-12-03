import numpy as np


class Ray:

    def __init__(self, pos=np.array([0, 0, 0, 1]), direction=np.array([0, 0, -1, 0])):
        self.pos = pos
        direction = direction / np.linalg.norm(direction)
        self.direction = direction

    def transform(self, mat):
        r = Ray()
        r.pos = np.matmul(mat, self.pos)
        r.direction = np.matmul(mat, self.direction)
        r.direction = r.direction / np.linalg.norm(r.direction)

        return r
