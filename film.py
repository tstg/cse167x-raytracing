import numpy as np
##import cv2
import copy
from PIL import Image


class Film:

    def __init__(self):
        self.image = np.zeros([100, 100, 3])
        self.filename = 'raytrace.png'

    def init_image(self, width=100, height=100):
        self.image = np.ones([height, width, 3])

    def commit(self, sample, color):
        row = int(sample[0])
        col = int(sample[1])
        self.image[row][col] = color

    def write_image(self):
        img = self.image

        img = img * 255
        img[img > 255] = 255

        img = np.uint8(img)
        i = Image.fromarray(img)

        if self.filename[-4:] != '.png':
            self.filename += '.png'

        i.save(self.filename)
