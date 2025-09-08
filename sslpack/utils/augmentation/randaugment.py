# Code in this file is adapted from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# This code is modified version of one of ildoonet, for randaugmentation of fixmatch.

import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw, PIL.Image
import numpy as np
from typing import Optional


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):
    return img.rotate(v)


def Sharpness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):
    if v <= 0.0:
        return img
    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


class RandAugment:
    """
    RandAugment data augmentation technique as described in <https://arxiv.org/abs/1909.13719>

    Args:
        num_ops (int):
            Number of augmentation transformations to apply sequentially. Expects a positive integer > 0. Defaults to 2.
        magnitude (int, optional):
            The magnitude for all the transformations. If unspecified, a magnitude is randomly chosen for each transformation. Expects an integer in [0, num_magnitude_bins-1] or None. Defaults to None.
        num_magnitude_bins (int):
            The number of discrete bins to use when choosing the magnitude for each transformation. Expects a positive integer > 1. Defaults to 31.
        use_cutout (bool):
            If True, Cutout is applied after the other augmentations. Defaults to False.
    """
    def __init__(self,
                 num_ops:int=2,
                 magnitude:Optional[int]=None,
                 num_magnitude_bins:int=31,
                 use_cutout:bool=False):

        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.use_cutout = use_cutout

        self.augments = [
            # (augment, min magnitude, max magnitude, signed)
            (Identity, 0, 1),
            (AutoContrast, 0, 1),
            (Brightness, 0.05, 0.95),
            (Color, 0.05, 0.95),
            (Contrast, 0.05, 0.95),
            (Equalize, 0, 1),
            (Posterize, 4, 8),
            (Rotate, -30, 30),
            (Sharpness, 0.05, 0.95),
            (ShearX, -0.3, 0.3),
            (ShearY, -0.3, 0.3),
            (Solarize, 0, 255),
            (TranslateX, -0.3, 0.3),
            (TranslateY, -0.3, 0.3),
        ]

        self.bw_augments = [
            (Brightness, 0.05, 0.95),
            (Equalize, 0, 1),
            (Identity, 0, 1),
            (Rotate, -30, 30),
            (Sharpness, 0.05, 0.95),
            (ShearX, -0.3, 0.3),
            (ShearY, -0.3, 0.3),
            (TranslateX, -0.3, 0.3),
            (TranslateY, -0.3, 0.3),
        ]

    def __call__(self, img:PIL.Image.Image):

        if img.mode == "L":
            augments = self.bw_augments
        elif img.mode == "RGB":
            augments = self.augments
        else:
            raise ValueError("Image mode should be either RGB or L.")
        ops = random.choices(augments, k=self.num_ops)

        for op, min_val, max_val in ops:
            if self.magnitude is None:
                mag = np.random.randint(0, self.num_magnitude_bins)
            else:
                mag = self.magnitude

            val = (float(mag) / float(self.num_magnitude_bins)) * float(max_val - min_val) + min_val

            img = op(img, val)

        if self.use_cutout is False:
            return img

        cutout_val = random.random() * 0.5
        return Cutout(img, cutout_val)  # for fixmatch
