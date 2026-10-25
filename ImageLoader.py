from PIL import Image

import numpy as np

class ImageLoader:
    @staticmethod
    def load(image_path):
        image = Image.open(image_path)
        image_gray = image.convert('L')
        np_arr = np.array(image_gray)
        res_image = np_arr.flatten().astype(np.float32) / 255.0

        return res_image