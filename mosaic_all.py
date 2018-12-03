import os
import numpy as np
from PIL import Image


def mosaic(root_dir, out_name):
    file_list = os.listdir(root_dir)
    res = np.ones(np.asarray(Image.open(os.path.join(root_dir, file_list[0]))).shape)
    for name in file_list:
        path = os.path.join(root_dir, name)
        img = np.asarray(Image.open(path))
        res = res * (img / 255)
    res = np.uint8(res * 255)
    img = Image.fromarray(res)
    img.save(out_name)
