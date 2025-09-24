from sslpack.datasets import SSLDataset
from torchvision.datasets import MNIST as MN
import torch
from torchvision import transforms
from sslpack.utils.data import TransformDataset, BasicDataset
from sslpack.utils.data import stratify_lbl_ulbl
from sslpack.utils.augmentation import RandAugment

from typing import Optional


class Mnist(SSLDataset):
    """
    MNIST is a handwriting image classification dataset.

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
        return_ulbl_labels (bool):
            If True, the labels for the unlabelled data are included. Defaults to False.
        return_idx (bool):
            If True, the indices are returned in the labelled and unlabelled datasets. Access them with key "idx". Defaults to False.
        crop_size (int):
            The length and width after cropping images during augmentations. Expects a positive integer > 0. Defaults to 28
        crop_ratio (float):
            The ratio used for padding when cropping during augmentations. Expects a float in [0,1]. Defaults to 0.875.
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
        seed: Optional[int] = None,
        return_idx: bool = False,
        return_ulbl_labels: bool = False,
        crop_size: int = 28,
        crop_ratio: float = 1,
        val_size: float = 1 / 6,
        download: bool = False,
    ):
        self.data_dir = data_dir
        self.lbls_per_class = lbls_per_class
        self.ulbls_per_class = ulbls_per_class
        self.val_per_class = val_per_class
        self.eval_per_class = eval_per_class
        self.seed = seed
        self.return_idx = return_idx
        self.return_ulbl_labels = return_ulbl_labels
        self.crop_size = crop_size
        self.crop_ratio = crop_ratio
        self.val_size = val_size
        self.download = download

        X_tr, y_tr, X_val, y_val, X_ts, y_ts = self._preprocess_data()
        self.X_mean, self.X_std = X_tr.mean(), X_tr.std()
        self._define_transforms()
        self._define_datasets(X_tr, y_tr, X_val, y_val, X_ts, y_ts)

    def _preprocess_data(self):
        mnist_tr = MN(root=self.data_dir, train=True, download=self.download)
        mnist_ts = MN(root=self.data_dir, train=False, download=self.download)

        num_val = int(self.val_size * len(mnist_tr))
        if self.seed is None:
            idx = torch.randperm(len(mnist_tr))
        else:
            idx = torch.randperm(
                len(mnist_tr), generator=torch.Generator().manual_seed(self.seed)
            )
        idx_val, idx_tr = idx[:num_val], idx[num_val:]

        X, y = mnist_tr.data.float() / 255, mnist_tr.targets
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_val, y_val = X[idx_val], y[idx_val]
        X_ts, y_ts = mnist_ts.data.float() / 255, mnist_ts.targets

        X_tr, X_ts, X_val = X_tr.unsqueeze(1), X_ts.unsqueeze(1), X_val.unsqueeze(1)

        return X_tr, y_tr, X_val, y_val, X_ts, y_ts

    def _define_transforms(self):
        self.weak_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.crop_size),
                transforms.RandomCrop(
                    self.crop_size,
                    padding=int(self.crop_size * (1 - self.crop_ratio)),
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.X_mean, self.X_std),
            ]
        )
        self.strong_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.crop_size),
                transforms.RandomCrop(
                    self.crop_size,
                    padding=int(self.crop_size * (1 - self.crop_ratio)),
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
                RandAugment(3),  # apply 3 augmentations
                transforms.ToTensor(),
                transforms.Normalize(self.X_mean, self.X_std),
            ]
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.crop_size),
                transforms.Normalize(self.X_mean, self.X_std),
            ]
        )

    def _define_datasets(self, X_tr, y_tr, X_val, y_val, X_ts, y_ts):
        X_tr_lb, y_tr_lb, X_tr_ulb, y_tr_ulb = stratify_lbl_ulbl(
            X=X_tr,
            y=y_tr,
            lbls_per_class=self.lbls_per_class,
            ulbls_per_class=self.ulbls_per_class,
            seed=self.seed,
        )

        if self.return_ulbl_labels is False:
            y_tr_ulb = None

        self.lbl_dataset = TransformDataset(
            X_tr_lb,
            y_tr_lb,
            transform=self.transform,
            weak_transform=self.weak_transform,
            strong_transform=self.strong_transform,
            return_idx=self.return_idx,
        )

        self.ulbl_dataset = TransformDataset(
            X_tr_ulb,
            y_tr_ulb,
            transform=self.transform,
            weak_transform=self.weak_transform,
            strong_transform=self.strong_transform,
            return_idx=self.return_idx,
        )

        if self.val_per_class is not None:
            # using this function to obtain a small validation set with labels per class
            X_val, y_val, _, _ = stratify_lbl_ulbl(
                X=X_val,
                y=y_val,
                lbls_per_class=self.val_per_class,
                seed=self.seed,
            )
        self.val_dataset = BasicDataset(
            X_val, y_val, transform=self.transform, return_idx=self.return_idx
        )

        if self.eval_per_class is not None:
            # using this function to obtain a small validation set with labels per class
            X_ts, y_ts, _, _ = stratify_lbl_ulbl(
                X=X_ts,
                y=y_ts,
                lbls_per_class=self.eval_per_class,
                seed=self.seed,
            )

        self.eval_dataset = BasicDataset(
            X_ts, y_ts, transform=self.transform, return_idx=self.return_idx
        )
