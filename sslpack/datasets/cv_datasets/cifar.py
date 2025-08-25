from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch
from PIL import Image

from sslpack.datasets import Dataset
from sslpack.utils.data import TransformDataset, BasicDataset, split_lb_ulb_balanced
from sslpack.utils.augmentation import RandAugment

import numpy as np


class Cifar(Dataset):

    def __init__(
        self,
        cifar,
        data_dir,
        lbls_per_class=4,
        ulbls_per_class=None,
        seed=None,
        crop_size=32,
        crop_ratio=1,
        download=True,
        return_ulbl_labels=False,
        return_idx=False,
    ):

        self.cifar = cifar
        self.return_ulbl_labels = return_ulbl_labels
        self.return_idx = return_idx

        self._define_transforms(crop_size, crop_ratio)

        self._get_dataset(lbls_per_class, ulbls_per_class, seed, data_dir, download)

    def _define_transforms(self, crop_size, crop_ratio):

        self.transform = transforms.Compose(
            [
                transforms.Resize(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.weak_transform = transforms.Compose(
            [
                transforms.Resize(crop_size),
                transforms.RandomCrop(
                    crop_size,
                    padding=int(crop_size * (1 - crop_ratio)),
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.strong_transform = transforms.Compose(
            [
                transforms.Resize(crop_size),
                transforms.RandomCrop(
                    crop_size,
                    padding=int(crop_size * (1 - crop_ratio)),
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
                RandAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _get_dataset(self, lbls_per_class, ulbls_per_class, seed, data_dir, download):

        train = self.cifar(data_dir, train=True, download=download)
        X_tr, y_tr = train.data, train.targets
        X_tr = [Image.fromarray(x) for x in X_tr]

        X_lb, y_lb, X_ulb, y_ulb = split_lb_ulb_balanced(
            X_tr,
            y_tr,
            lbls_per_class=lbls_per_class,
            ulbls_per_class=ulbls_per_class,
            seed=seed,
        )

        if self.return_ulbl_labels is False:
            y_ulb = None

        self.lbl_dataset = TransformDataset(
            X=X_lb,
            y=y_lb,
            transform=self.transform,
            weak_transform=self.weak_transform,
            strong_transform=self.strong_transform,
            return_idx=self.return_idx
        )

        self.ulbl_dataset = TransformDataset(
            X=X_ulb,
            y=y_ulb,
            transform=self.transform,
            weak_transform=self.weak_transform,
            strong_transform=self.strong_transform,
            return_idx=self.return_idx
        )

        test = self.cifar(data_dir, train=False, download=download)
        X_ts, y_ts = test.data, test.targets
        X_ts = [Image.fromarray(x) for x in X_ts]

        self.eval_dataset = BasicDataset(X=X_ts, y=y_ts, transform=self.transform, return_idx=self.return_idx)


class Cifar10(Cifar):
    """
    A Cifar10 semi supervised learning dataset with transformations.

    """

    def __init__(
        self,
        lbls_per_class=4,
        ulbls_per_class=None,
        seed=None,
        crop_size=32,
        crop_ratio=1,
        return_ulb_labels=False,
        return_idx=False,
        data_dir="~/.sslpack/datasets/CIFAR10",
        download=True,
    ):
        """
        Initialise a CIFAR100 dataset.  Contains a labelled, unlabelled and evaluation dataset.

        Elements from the datasets are return as dictionaries with keys
            "X": The original features as a tensor
            "weak": The weak augmentation applied to the features
            "strong": The strong augmentation applied to the features
            "y": The labels, if applicable
            "idx": The dataset index, if enabled.

        Args:
            lbls_per_class: The number of labelled observations to include per class
            ulbls_per_class: The number of unlabelled observations to include per class. By default all remaining unlabelled observations are used
            seed: The seed for randomly choosing the labelled instances
            crop_size: The length/width of crop size for resizing (square) during augmentations
            crop_ratio: The crop ratio used for padding when cropping during augmentations
            return_ulb_labels: If true, the labels for the unlabelled data are included
            return_idx: If true, the indices are returned when accessing a dataset
            data_dir: The directory to save the dataset
            download: If true, the dataset is downloaded if it does not already exist
        """
        super().__init__(
            CIFAR10,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            seed,
            crop_size,
            crop_ratio,
            download,
            return_ulb_labels,
            return_idx
        )


class Cifar100(Cifar):
    """
    A Cifar100 semi supervised learning dataset with transformations
    """

    def __init__(
        self,
        lbls_per_class=4,
        ulbls_per_class=None,
        seed=None,
        crop_size=32,
        crop_ratio=1,
        return_ulbl_labels=False,
        return_idx=False,
        data_dir="~/.sslpack/datasets/CIFAR100",
        download=True,
    ):
        """

        Initialise a CIFAR100 dataset.  Contains a labelled, unlabelled and evaluation dataset.

        Elements from the datasets are return as dictionaries with keys
            "X": The original features as a tensor
            "weak": The weak augmentation applied to the features
            "strong": The strong augmentation applied to the features
            "y": The labels, if applicable
            "idx": The dataset index, if enabled.

        Args:
            lbls_per_class: The number of labelled observations to include per class
            ulbls_per_class: The number of unlabelled observations to include per class. By default all remaining unlabelled observations are used
            seed: The seed for randomly choosing the labelled instances
            crop_size: The length/width of crop size for resizing (square) during augmentations
            crop_ratio: The crop ratio used for padding when cropping during augmentations
            return_ulb_labels: If true, the labels for the unlabelled data are included
            return_idx: If true, the indices are returned when accessing a dataset
            data_dir: The directory to save the dataset
            download: If true, the dataset is downloaded if it does not already exist
        """
        super().__init__(
            CIFAR100,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            seed,
            crop_size,
            crop_ratio,
            download,
            return_ulbl_labels,
            return_idx
        )
