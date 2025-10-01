import random
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import numpy as np
from typing import Optional


def Identity(img, v):
    return img


def AutoContrast(img, _):
    return F.autocontrast(img)


def Brightness(img, v):
    return F.adjust_brightness(img, v)


def Color(img, v):
    return F.adjust_saturation(img, v)


def Contrast(img, v):
    return F.adjust_contrast(img, v)


def Equalize(img, _):
    return F.equalize(img)


def Invert(img, _):
    return F.invert(img)


def Posterize(img, v):
    v = int(v)
    v = max(1, v)
    return F.posterize(img, v)


def Rotate(img, v):
    return F.rotate(img, v, interpolation=T.InterpolationMode.BILINEAR, expand=False)


def Sharpness(img, v):
    return F.adjust_sharpness(img, v)


def ShearX(img, v):
    return F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[v, 0])


def ShearY(img, v):
    return F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[0, v])


def TranslateX(img, v):
    width = F.get_image_size(img)[1]
    v = int(v * width)
    return F.affine(img, angle=0, translate=[v, 0], scale=1.0, shear=[0.0, 0.0])


def TranslateXabs(img, v):
    return F.affine(img, angle=0, translate=[int(v), 0], scale=1.0, shear=[0.0, 0.0])


def TranslateY(img, v):
    height = F.get_image_size(img)[0]
    v = int(v * height)
    return F.affine(img, angle=0, translate=[0, v], scale=1.0, shear=[0.0, 0.0])


def TranslateYabs(img, v):
    return F.affine(img, angle=0, translate=[0, int(v)], scale=1.0, shear=[0.0, 0.0])


def Solarize(img, v):
    v = int(v)
    return F.solarize(img, v)


def Cutout(img, v):
    if v <= 0.0:
        return img
    h, w = F.get_image_size(img)
    v = int(v * w)
    return CutoutAbs(img, v)


def CutoutAbs(img, v):
    if v <= 0:
        return img

    h, w = F.get_image_size(img)
    x0 = int(np.random.uniform(0, w))
    y0 = int(np.random.uniform(0, h))
    x0 = max(0, x0 - v // 2)
    y0 = max(0, y0 - v // 2)
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    # Create a tensor mask
    if isinstance(img, torch.Tensor):
        mask_color = torch.tensor([125/255.0, 123/255.0, 114/255.0], dtype=img.dtype, device=img.device)
        if img.shape[0] == 1:  # Grayscale
            mask_color = mask_color[0:1]
        img[..., y0:y1, x0:x1] = mask_color[:, None, None]
        return img
    else:
        raise TypeError("Expected image to be a torch.Tensor")


class RandAugmentV2:
    """
    RandAugment implemented with torchvision.transforms.v2 (tensor-based).
    Works on torch tensors of shape [C, H, W].
    """

    def __init__(self,
                 num_ops: int = 2,
                 magnitude: Optional[int] = None,
                 num_magnitude_bins: int = 31,
                 use_cutout: bool = False):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.use_cutout = use_cutout

        self.augments = [
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

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor image [C, H, W]")

        if img.ndim != 3 or img.shape[0] not in [1, 3]:
            raise ValueError("Image must be of shape [C, H, W] with 1 or 3 channels")

        is_bw = img.shape[0] == 1
        augments = self.bw_augments if is_bw else self.augments

        ops = random.choices(augments, k=self.num_ops)

        for op, min_val, max_val in ops:
            if self.magnitude is None:
                mag = np.random.randint(0, self.num_magnitude_bins)
            else:
                mag = self.magnitude

            val = (float(mag) / float(self.num_magnitude_bins)) * (max_val - min_val) + min_val
            img = op(img, val)

        if self.use_cutout:
            cutout_val = random.uniform(0, 0.5)
            img = Cutout(img, cutout_val)

        return img
