import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from PIL import Image

from sslpack.datasets import SSLDataset
from sslpack.utils.data import TransformDataset, BasicDataset, stratify_lbl_ulbl
from sslpack.utils.augmentation import RandAugment

from typing import Optional


class Cifar(SSLDataset):

    def __init__(
        self,
        cifar,
        data_dir,
        lbls_per_class=4,
        ulbls_per_class=None,
        val_per_class=None,
        eval_per_class=None,
        seed=None,
        crop_size=32,
        crop_ratio=1,
        download=True,
        return_ulbl_labels=False,
        return_idx=False,
        val_size=1 / 5,
    ):

        self.cifar = cifar
        self.return_ulbl_labels = return_ulbl_labels
        self.return_idx = return_idx
        self.val_size = val_size
        self.val_per_class = val_per_class
        self.eval_per_class = eval_per_class

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
                RandAugment(3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _get_dataset(self, lbls_per_class, ulbls_per_class, seed, data_dir, download):

        train = self.cifar(data_dir, train=True, download=download)
        num_val = int(self.val_size * len(train))
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train), generator=generator)
        idx_val, idx_tr = idx[:num_val], idx[num_val:]

        X, y = train.data, torch.tensor(train.targets)
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_val, y_val = X[idx_val], y[idx_val]
        X_tr = [Image.fromarray(x) for x in X_tr]
        X_val = [Image.fromarray(x) for x in X_val]

        X_lb, y_lb, X_ulb, y_ulb = stratify_lbl_ulbl(
            X=X_tr,
            y=y_tr,
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
            return_idx=self.return_idx,
        )

        self.ulbl_dataset = TransformDataset(
            X=X_ulb,
            y=y_ulb,
            transform=self.transform,
            weak_transform=self.weak_transform,
            strong_transform=self.strong_transform,
            return_idx=self.return_idx,
        )

        test = self.cifar(data_dir, train=False, download=download)
        X_ts, y_ts = test.data, torch.tensor(test.targets)
        X_ts = [Image.fromarray(x) for x in X_ts]

        if self.val_per_class is not None:
            # using this function to obtain a small validation set with labels per class
            X_val, y_val, _, _ = stratify_lbl_ulbl(
                X=X_val,
                y=y_val,
                lbls_per_class=self.val_per_class,
                seed=self.seed,
            )

        self.val_dataset = BasicDataset(
            X=X_val, y=y_val, transform=self.transform, return_idx=self.return_idx
        )

        self.eval_dataset = BasicDataset(
            X=X_ts, y=y_ts, transform=self.transform, return_idx=self.return_idx
        )

        if self.eval_per_class is not None:
            # using this function to obtain a small validation set with labels per class
            X_ts, y_ts, _, _ = stratify_lbl_ulbl(
                X=X_ts,
                y=y_ts,
                lbls_per_class=self.eval_per_class,
                seed=self.seed,
            )


class Cifar10(Cifar):
    """
    CIFAR10 is an image classification dataset.

    There are 60000 total examples with 10 classes, and all features are normalised.

    The train/test split is 50000/10000
    The training data is split into labelled and unlabelled parts.

    Contains an labelled and unlabelled set for training, a validation set and an evaluation set.
    Elements from these datasets are returned as dictionaries. See `sslpack.datasets.SSLDataset`.
    Supports data augmentation. Weak transformations are random horizontal flips.
    Strong transforms are done with `sslpack.utils.augmentation.RandAugment`

    Args:
        data_dir (str):
            The directory where the data is saved, or where it will be saved to if download=True
        lbls_per_class (int):
            The number of labelled observations to include per class. Expects a positive integer > 0
        ulbls_per_class (int, optional):
            The number of unlabelled observations to include per class. If unspecified, all
            remaining examples after selecting the labelled examples are used. By default unspecified.
        val_per_class (int, optional):
            The number of observations per class to use in the validation set. If unspecified, the entire
            validation set is used. By default unspecified. Expects an integer > 0
        eval_per_class (int, optional):
            The number of observations per class to use in the evaluation set. If unspecified, the entire
            evaluation set is used. By default unspecified. Expects an integer > 0
        seed (int, optional):
            The seed for randomly choosing the labelled instances
        crop_size (int):
            The length and width after cropping images during augmentations. Expects a positive integer > 0. Defaults to 28
        crop_ratio (float):
            The ratio used for padding when cropping during augmentations. Expects a float in [0,1]. Defaults to 0.875.
        return_ulbl_labels (bool):
            If True, the labels for the unlabelled data are included. Defaults to False.
        return_idx (bool):
            If True, the indices are returned in the labelled and unlabelled datasets. Access them with key "idx". Defaults to False.
        val_size (float):
            The proportion of the training data to use as the validation set. Defaults to 1/5
        download (bool):
            If True, the dataset is downloaded if it does not already exist in the specified directory.
            If False, an error will occur unless the dataset already exists. Defaults to False.
    """

    def __init__(
        self,
        data_dir: str,
        lbls_per_class: int,
        ulbls_per_class: Optional[int] = None,
        val_per_class: Optional[int] = None,
        eval_per_class: Optional[int] = None,
        return_ulbl_labels: bool = False,
        return_idx: bool = False,
        seed: Optional[int] = None,
        crop_size: int = 32,
        crop_ratio: float = 1,
        val_size: float = 1 / 5,
        download: bool = False,
    ):
        """
        Initialise a CIFAR100 SSL dataset.
        """
        super().__init__(
            CIFAR10,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            val_per_class,
            eval_per_class,
            seed,
            crop_size,
            crop_ratio,
            download,
            return_ulbl_labels,
            return_idx,
            val_size,
        )


class Cifar100(Cifar):
    """
    CIFAR100 is an image classification dataset.

    There are 60000 total examples with 100 classes, and all features are normalised.

    The train/test split is 50000/10000
    The training data is split into labelled and unlabelled parts.

    Contains an labelled and unlabelled set for training, a validation set and an evaluation set.
    Elements from these datasets are returned as dictionaries. See `sslpack.datasets.SSLDataset`.
    Supports data augmentation. Weak transformations are random horizontal flips.
    Strong transforms are done with `sslpack.utils.augmentation.RandAugment`

    Args:
        data_dir (str):
            The directory where the data is saved, or where it will be saved to if download=True
        lbls_per_class (int):
            The number of labelled observations to include per class. Expects a positive integer > 0
        ulbls_per_class (int, optional):
            The number of unlabelled observations to include per class. If unspecified, all
            remaining examples after selecting the labelled examples are used. By default unspecified.
        val_per_class (int, optional):
            The number of observations per class to use in the validation set. If unspecified, the entire
            validation set is used. By default unspecified. Expects an integer > 0
        eval_per_class (int, optional):
            The number of observations per class to use in the evaluation set. If unspecified, the entire
            evaluation set is used. By default unspecified. Expects an integer > 0
        seed (int, optional):
            The seed for randomly choosing the labelled instances
        crop_size (int):
            The length and width after cropping images during augmentations. Expects a positive integer > 0. Defaults to 28
        crop_ratio (float):
            The ratio used for padding when cropping during augmentations. Expects a float in [0,1]. Defaults to 0.875.
        return_ulbl_labels (bool):
            If True, the labels for the unlabelled data are included. Defaults to False.
        return_idx (bool):
            If True, the indices are returned in the labelled and unlabelled datasets. Access them with key "idx". Defaults to False.
        val_size (float):
            The proportion of the training data to use as the validation set. Defaults to 1/5
        download (bool):
            If True, the dataset is downloaded if it does not already exist in the specified directory.
            If False, an error will occur unless the dataset already exists. Defaults to False.
    """

    def __init__(
        self,
        data_dir: str,
        lbls_per_class: int,
        ulbls_per_class: Optional[int] = None,
        val_per_class: Optional[int] = None,
        eval_per_class: Optional[int] = None,
        return_ulbl_labels: bool = False,
        return_idx: bool = False,
        seed: Optional[int] = None,
        crop_size: int = 32,
        crop_ratio: float = 1,
        val_size: float = 1 / 5,
        download: bool = False,
    ):
        """

        Initialise a CIFAR100 SSL dataset.
        """
        super().__init__(
            CIFAR100,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            val_per_class,
            eval_per_class,
            seed,
            crop_size,
            crop_ratio,
            download,
            return_ulbl_labels,
            return_idx,
            val_size,
        )
