import numpy as np


class TileSampler:

    def __init__(self, left=0, top=0, tile_width=32, tile_height=32):
        self.left = left
        self.top = top
        self.tile_width = tile_width
        self.tile_height = tile_height

    def get_sample(self):
        left = self.left
        top = self.top
        tile_width = self.tile_width
        tile_height = self.tile_height

        for i in range(top, top + tile_height):
            for j in range(left, left + tile_width):
                yield np.array([i + 0.5, j + 0.5])
