import numpy as np
from PIL import Image
from skimage.util import img_as_float, img_as_ubyte, random_noise


class NoiseAugmentation(object):
    def __init__(self, mode='gaussian', mean=0.0, var=0.01, density=0.05, proportion=0.5, seed=None):
        self.mode = mode
        self.density = density
        self.proportion = proportion
        self.mean = mean
        self.var = var
        self.seed = seed

    def __call__(self, img):
        if self.seed is not None:
            np.random.seed(self.seed)

        img_np = np.array(img)
        img_float = img_as_float(img_np)

        if self.mode == 's&p':
            noisy_img = random_noise(img_float, mode=self.mode, amount=self.density, salt_vs_pepper=self.proportion)
        elif self.mode == 'speckle':
            noisy_img = random_noise(img_float, mode='speckle', var=self.var)
        elif self.mode == 'poisson':
            noisy_img = random_noise(img_float, mode='poisson')
        else:
            noisy_img = random_noise(img_float, mode='gaussian', mean=self.mean, var=self.var)

        img_ubyte = img_as_ubyte(noisy_img)
        return Image.fromarray(img_ubyte)
