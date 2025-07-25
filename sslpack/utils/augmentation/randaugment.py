# Code in this file is adapted from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# This code is modified version of one of ildoonet, for randaugmentation of fixmatch.

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
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
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
    ]
    return l


def augment_list_no_color():
    l = [
        # (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        # (Color, 0.05, 0.95),
        # (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        # (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        # (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
    ]
    return l


class RandAugment:
    def __init__(self, n, m, exclude_color_aug=False, bw=False):
        self.n = n
        self.m = m
        self.bw = bw
        if not exclude_color_aug:
            self.augment_list = augment_list()
        else:
            self.augment_list = augment_list_no_color()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        if self.bw is True:
            return img

        cutout_val = random.random() * 0.5
        return Cutout(img, cutout_val)  # for fixmatch
