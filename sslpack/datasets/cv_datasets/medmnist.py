from sslpack.datasets import SSLDataset
import torch
from torchvision import transforms

# from torchvision.transforms import RandAugment
from sslpack.utils.data import TransformDataset, BasicDataset
from sslpack.utils.data import split_lbl_ulbl
from sslpack.utils.augmentation import RandAugment

import os
from typing import Optional


class MedMnist(SSLDataset):

    def __init__(
        self,
        medmnist_class,
        data_dir,
        lbls_per_class,
        ulbls_per_class=None,
        val_per_class=None,
        eval_per_class=None,
        img_size=28,
        seed=None,
        return_idx=False,
        return_ulbl_labels=False,
        crop_size=28,
        crop_ratio=1,
        num_augment=3,
        mag_augment=5,
        download=False,
    ):

        self.medmnist_class = medmnist_class
        self.data_dir = data_dir
        self.lbls_per_class = lbls_per_class
        self.ulbls_per_class = ulbls_per_class
        self.val_per_class = val_per_class
        self.eval_per_class = eval_per_class
        self.seed = seed
        self.img_size = img_size
        self.return_idx = return_idx
        self.return_ulbl_labels = return_ulbl_labels
        self.crop_size = crop_size
        self.crop_ratio = crop_ratio
        self.num_augment = num_augment
        self.mag_augment = mag_augment
        self.download = download

        os.makedirs(data_dir, exist_ok=True)
        X_tr, y_tr, X_val, y_val, X_ts, y_ts = self._preprocess_data()
        self.X_mean, self.X_std = X_tr.mean(), X_tr.std()
        self._define_transforms()
        self._define_datasets(X_tr, y_tr, X_val, y_val, X_ts, y_ts)

    def _check_import(self):
        try:
            import medmnist
        except ImportError as e:
            raise ImportError(
                "This dataset requires `medmnist`. Install it with:\n"
                "    pip install sslpack[medmnist]"
            ) from e

    def _preprocess_data(self):

        tr = self.medmnist_class(
            root=self.data_dir,
            split="train",
            download=self.download,
            size=self.img_size,
        )
        ts = self.medmnist_class(
            root=self.data_dir, split="test", download=self.download, size=self.img_size
        )
        val = self.medmnist_class(
            root=self.data_dir, split="val", download=self.download, size=self.img_size
        )

        X_tr, y_tr = torch.tensor(tr.imgs).float() / 255, torch.tensor(tr.labels)
        X_ts, y_ts = torch.tensor(ts.imgs).float() / 255, torch.tensor(ts.labels)
        X_val, y_val = torch.tensor(val.imgs).float() / 255, torch.tensor(val.labels)

        # different medmnist datasets have different channel configuration
        X_tr = self._permute(X_tr)
        X_ts = self._permute(X_ts)
        X_val = self._permute(X_val)

        y_tr, y_ts, y_val = y_tr.squeeze(1), y_ts.squeeze(1), y_val.squeeze(1)

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
                RandAugment(self.num_augment, self.mag_augment),
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
        X_tr_lb, y_tr_lb, X_tr_ulb, y_tr_ulb = split_lbl_ulbl(
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
            X_val, y_val, _, _ = split_lbl_ulbl(
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
            X_ts, y_ts, _, _ = split_lbl_ulbl(
                X=X_ts,
                y=y_ts,
                lbls_per_class=self.eval_per_class,
                seed=self.seed,
            )

        self.eval_dataset = BasicDataset(
            X_ts, y_ts, transform=self.transform, return_idx=self.return_idx
        )

    def _permute(self, X):
        return X.permute(0, 3, 1, 2)


class BloodMnist(MedMnist):
    """
    BloodMNIST is a blood cell medical imaging dataset for image classification.

    There are 17,092 total examples with 8 classes, and all features are normalised.

    The train/val/test split is 11,959 / 1,712 / 3,421.
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
        img_size (int):
            The image size of the dataset. Defaults to the 28x28 version of the dataset.
            Specify 64, 128 or 224 for MedMNIST+ datasets.
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
        num_augment (int):
            The number of RandAugments to apply for strong augmentation. Expects a positive integer >= 0. Defaults to 3.
        mag_augment (int):
            The magnitude of RandAugments to apply for strong augmentation. Expects a positive integer > 0. Defaults to 5.
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
        img_size: int = 28,
        seed: Optional[int] = None,
        return_idx: bool = False,
        return_ulbl_labels: bool = False,
        crop_size: int = 28,
        crop_ratio: float = 1,
        num_augment: int = 3,
        mag_augment: int = 5,
        download: bool = True,
    ):
        self._check_import()
        from medmnist import BloodMNIST

        super().__init__(
            BloodMNIST,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            val_per_class,
            eval_per_class,
            img_size,
            seed,
            return_idx,
            return_ulbl_labels,
            crop_size,
            crop_ratio,
            num_augment,
            mag_augment,
            download,
        )


class PathMnist(MedMnist):
    """
    PathMNIST is a colon pathology medical imaging dataset for image classification.

    There are 107,180 total examples with 9 classes, and all features are normalised.

    The train/val/test split is 89,996 / 10,004 / 7,180.
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
        img_size (int):
            The image size of the dataset. Defaults to the 28x28 version of the dataset.
            Specify 64, 128 or 224 for MedMNIST+ datasets.
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
        num_augment (int):
            The number of RandAugments to apply for strong augmentation. Expects a positive integer >= 0. Defaults to 3.
        mag_augment (int):
            The magnitude of RandAugments to apply for strong augmentation. Expects a positive integer > 0. Defaults to 5.
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
        img_size: int = 28,
        seed: Optional[int] = None,
        return_idx: bool = False,
        return_ulbl_labels: bool = False,
        crop_size: int = 28,
        crop_ratio: float = 1,
        num_augment: int = 3,
        mag_augment: int = 5,
        download: bool = True,
    ):

        self._check_import()
        from medmnist import PathMNIST

        super().__init__(
            PathMNIST,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            val_per_class,
            eval_per_class,
            img_size,
            seed,
            return_idx,
            return_ulbl_labels,
            crop_size,
            crop_ratio,
            num_augment,
            mag_augment,
            download,
        )


class ChestMnist(MedMnist):
    """
    ChestMNIST is a chest x-ray medical imaging dataset for image classification.

    There are 112,120 total examples with 14 classes, and all features are normalised.

    The train/val/test split is 78,468 / 11,219 / 22,433.
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
        img_size (int):
            The image size of the dataset. Defaults to the 28x28 version of the dataset.
            Specify 64, 128 or 224 for MedMNIST+ datasets.
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
        num_augment (int):
            The number of RandAugments to apply for strong augmentation. Expects a positive integer >= 0. Defaults to 3.
        mag_augment (int):
            The magnitude of RandAugments to apply for strong augmentation. Expects a positive integer > 0. Defaults to 5.
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
        img_size: int = 28,
        seed: Optional[int] = None,
        return_idx: bool = False,
        return_ulbl_labels: bool = False,
        crop_size: int = 28,
        crop_ratio: float = 1,
        num_augment: int = 3,
        mag_augment: int = 5,
        download: bool = True,
    ):
        self._check_import()
        from medmnist import ChestMNIST

        super().__init__(
            ChestMNIST,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            val_per_class,
            eval_per_class,
            img_size,
            seed,
            return_idx,
            return_ulbl_labels,
            crop_size,
            crop_ratio,
            num_augment,
            mag_augment,
            download,
        )


class DermaMnist(MedMnist):
    """
    DermaMNIST is a dermatoscope medical imaging dataset for image classification.

    There are 10,015 total examples with 14 classes, and all features are normalised.

    The train/val/test split is 7,007 / 1,003 / 2,005.
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
        img_size (int):
            The image size of the dataset. Defaults to the 28x28 version of the dataset.
            Specify 64, 128 or 224 for MedMNIST+ datasets.
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
        num_augment (int):
            The number of RandAugments to apply for strong augmentation. Expects a positive integer >= 0. Defaults to 3.
        mag_augment (int):
            The magnitude of RandAugments to apply for strong augmentation. Expects a positive integer > 0. Defaults to 5.
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
        img_size: int = 28,
        seed: Optional[int] = None,
        return_idx: bool = False,
        return_ulbl_labels: bool = False,
        crop_size: int = 28,
        crop_ratio: float = 1,
        num_augment: int = 3,
        mag_augment: int = 5,
        download: bool = True,
    ):
        self._check_import()
        from medmnist import DermaMNIST

        super().__init__(
            DermaMNIST,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            val_per_class,
            eval_per_class,
            img_size,
            seed,
            return_idx,
            return_ulbl_labels,
            crop_size,
            crop_ratio,
            num_augment,
            mag_augment,
            download,
        )


class BreastMnist(MedMnist):
    """
    BreastMNIST is a breast ultrasound medical imaging dataset for image classification.

    There are 780 total examples with 2 classes, and all features are normalised.

    The train/val/test split is 546 / 78 / 156.
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
        img_size (int):
            The image size of the dataset. Defaults to the 28x28 version of the dataset.
            Specify 64, 128 or 224 for MedMNIST+ datasets.
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
        num_augment (int):
            The number of RandAugments to apply for strong augmentation. Expects a positive integer >= 0. Defaults to 3.
        mag_augment (int):
            The magnitude of RandAugments to apply for strong augmentation. Expects a positive integer > 0. Defaults to 5.
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
        img_size: int = 28,
        seed: Optional[int] = None,
        return_idx: bool = False,
        return_ulbl_labels: bool = False,
        crop_size: int = 28,
        crop_ratio: float = 1,
        num_augment: int = 3,
        mag_augment: int = 5,
        download: bool = True,
    ):
        self._check_import()
        from medmnist import BreastMNIST

        super().__init__(
            BreastMNIST,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            val_per_class,
            eval_per_class,
            img_size,
            seed,
            return_idx,
            return_ulbl_labels,
            crop_size,
            crop_ratio,
            num_augment,
            mag_augment,
            download,
        )

    def _permute(self, X):
        return X.permute(0, 1, 2)
