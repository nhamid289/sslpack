from sslpack.datasets import Dataset
import torch
from torchvision import transforms
from sslpack.utils.data import TransformDataset, BasicDataset
from sslpack.utils.data import split_lb_ulb_balanced
from sslpack.utils.augmentation import RandAugment

import os


class MedMnist(Dataset):

    def __init__(
        self,
        medmnist,
        data_dir,
        lbls_per_class,
        ulbls_per_class=None,
        img_size=28,
        seed=None,
        return_idx=False,
        return_ulbl_labels=False,
        crop_size=28,
        crop_ratio=1,
        download=False,
    ):

        os.makedirs(data_dir, exist_ok=True)

        tr = medmnist(root=data_dir, split="train", download=download, size=img_size)
        ts = medmnist(root=data_dir, split="test", download=download, size=img_size)
        val = medmnist(root=data_dir, split="val", download=download, size=img_size)

        X_tr, y_tr = torch.tensor(tr.imgs).float() / 255, torch.tensor(tr.labels)
        X_ts, y_ts = torch.tensor(ts.imgs).float() / 255, torch.tensor(ts.labels)
        X_val, y_val = torch.tensor(val.imgs).float() / 255, torch.tensor(val.labels)

        # different medmnist datasets have different channel configuration
        X_tr = self._permute(X_tr)
        X_ts = self._permute(X_ts)
        X_val = self._permute(X_val)

        y_tr, y_ts, y_val = y_tr.squeeze(1), y_ts.squeeze(1), y_val.squeeze(1)

        self.weak_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(crop_size),
                transforms.RandomCrop(
                    crop_size,
                    padding=int(crop_size * (1 - crop_ratio)),
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(X_tr.mean(), X_tr.std()),
            ]
        )
        self.strong_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(crop_size),
                transforms.RandomCrop(
                    crop_size,
                    padding=int(crop_size * (1 - crop_ratio)),
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
                RandAugment(3, 5, exclude_color_aug=True, bw=True),
                transforms.ToTensor(),
                transforms.Normalize(X_tr.mean(), X_tr.std()),
            ]
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(crop_size),
                transforms.Normalize(X_tr.mean(), X_tr.std()),
            ]
        )

        X_tr_lb, y_tr_lb, X_tr_ulb, y_tr_ulb = split_lb_ulb_balanced(
            X=X_tr,
            y=y_tr,
            lbls_per_class=lbls_per_class,
            ulbls_per_class=ulbls_per_class,
            seed=seed,
        )

        if not return_ulbl_labels:
            y_tr_ulb = None

        self.lbl_dataset = TransformDataset(
            X_tr_lb,
            y_tr_lb,
            transform=self.transform,
            weak_transform=self.weak_transform,
            strong_transform=self.strong_transform,
            return_idx=return_idx,
        )

        self.ulbl_dataset = TransformDataset(
            X_tr_ulb,
            y_tr_ulb,
            transform=self.transform,
            weak_transform=self.weak_transform,
            strong_transform=self.strong_transform,
            return_idx=return_idx,
        )

        self.val_dataset = BasicDataset(
            X_val, y_val, transform=self.transform, return_idx=return_idx
        )
        self.eval_dataset = BasicDataset(
            X_ts, y_ts, transform=self.transform, return_idx=return_idx
        )

    def _permute(self, X):
        return X.permute(0, 3, 1, 2)

class BloodMnist(MedMnist):

    def __init__(
        self,
        data_dir,
        lbls_per_class,
        ulbls_per_class=None,
        img_size=28,
        seed=None,
        return_idx=False,
        return_ulbl_labels=False,
        crop_size=28,
        crop_ratio=1,
        download=True,
    ):
        """
        Initialise an BloodMNIST SSL dataset.  Contains a labelled, unlabelled , validation, evaluation dataset. All features are normalised.

        Elements from the datasets are return as dictionaries with keys
            "X": The original features as a tensor
            "weak": The weak augmentation applied to the features
            "strong": The strong augmentation applied to the features
            "y": The labels, if applicable
            "idx": The dataset index. This key is only returned if return_ulbl_labels=True

        Args:
            data_dir: The directory where the data is saved, or where it will be saved to if download=True
            lbls_per_class: The number of labelled observations to include per class
            ulbls_per_class: The number of unlabelled observations to include per class. By default all remaining unlabelled observations are used
            img_size: By default, use MedMNIST with 28x28. Specify 64, 128 or 224 for MedMNIST+ datasets.
            seed: The seed for randomly choosing the labelled instances
            crop_size: The length/width of crop size for resizing (square) during augmentations
            crop_ratio: The crop ratio used for padding when cropping during augmentations
            return_ulbl_labels: If true, the labels for the unlabelled data are included
            return_idx: If true, the indices are returned when accessing a dataset
            download: If true, the dataset is downloaded if it does not already exist
        """
        try:
            from medmnist import BloodMNIST
        except ImportError as e:
            raise ImportError(
                "This dataset requires `medmnist`. Install it with:\n"
                "    pip install sslpack[medmnist]"
            ) from e

        super().__init__(
            BloodMNIST,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            img_size,
            seed,
            return_idx,
            return_ulbl_labels,
            crop_size,
            crop_ratio,
            download,
        )


class PathMnist(MedMnist):

    def __init__(
        self,
        data_dir,
        lbls_per_class,
        ulbls_per_class=None,
        img_size=28,
        seed=None,
        return_idx=False,
        return_ulbl_labels=False,
        crop_size=28,
        crop_ratio=1,
        download=True,
    ):
        """
        Initialise an PathMNIST SSL dataset.  Contains a labelled, unlabelled , validation, evaluation dataset. All features are normalised.

        Elements from the datasets are return as dictionaries with keys
            "X": The original features as a tensor
            "weak": The weak augmentation applied to the features
            "strong": The strong augmentation applied to the features
            "y": The labels, if applicable
            "idx": The dataset index. This key is only returned if return_ulbl_labels=True

        Args:
            data_dir: The directory where the data is saved, or where it will be saved to if download=True
            lbls_per_class: The number of labelled observations to include per class
            ulbls_per_class: The number of unlabelled observations to include per class. By default all remaining unlabelled observations are used
            img_size: By default, use MedMNIST with 28x28. Specify 64, 128 or 224 for MedMNIST+ datasets.
            seed: The seed for randomly choosing the labelled instances
            crop_size: The length/width of crop size for resizing (square) during augmentations
            crop_ratio: The crop ratio used for padding when cropping during augmentations
            return_ulbl_labels: If true, the labels for the unlabelled data are included
            return_idx: If true, the indices are returned when accessing a dataset
            download: If true, the dataset is downloaded if it does not already exist
        """
        try:
            from medmnist import PathMNIST
        except ImportError as e:
            raise ImportError(
                "This dataset requires `medmnist`. Install it with:\n"
                "    pip install sslpack[medmnist]"
            ) from e

        super().__init__(
            PathMNIST,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            img_size,
            seed,
            return_idx,
            return_ulbl_labels,
            crop_size,
            crop_ratio,
            download,
        )


class ChestMnist(MedMnist):

    def __init__(
        self,
        data_dir,
        lbls_per_class,
        ulbls_per_class=None,
        img_size=28,
        seed=None,
        return_idx=False,
        return_ulbl_labels=False,
        crop_size=28,
        crop_ratio=1,
        download=True,
    ):
        """
        Initialise an ChestMNIST SSL dataset.  Contains a labelled, unlabelled , validation, evaluation dataset. All features are normalised.

        Elements from the datasets are return as dictionaries with keys
            "X": The original features as a tensor
            "weak": The weak augmentation applied to the features
            "strong": The strong augmentation applied to the features
            "y": The labels, if applicable
            "idx": The dataset index. This key is only returned if return_ulbl_labels=True

        Args:
            data_dir: The directory where the data is saved, or where it will be saved to if download=True
            lbls_per_class: The number of labelled observations to include per class
            ulbls_per_class: The number of unlabelled observations to include per class. By default all remaining unlabelled observations are used
            img_size: By default, use MedMNIST with 28x28. Specify 64, 128 or 224 for MedMNIST+ datasets.
            seed: The seed for randomly choosing the labelled instances
            crop_size: The length/width of crop size for resizing (square) during augmentations
            crop_ratio: The crop ratio used for padding when cropping during augmentations
            return_ulbl_labels: If true, the labels for the unlabelled data are included
            return_idx: If true, the indices are returned when accessing a dataset
            download: If true, the dataset is downloaded if it does not already exist
        """
        try:
            from medmnist import ChestMNIST
        except ImportError as e:
            raise ImportError(
                "This dataset requires `medmnist`. Install it with:\n"
                "    pip install sslpack[medmnist]"
            ) from e

        super().__init__(
            ChestMNIST,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            img_size,
            seed,
            return_idx,
            return_ulbl_labels,
            crop_size,
            crop_ratio,
            download,
        )


class DermaMnist(MedMnist):

    def __init__(
        self,
        data_dir,
        lbls_per_class,
        ulbls_per_class=None,
        img_size=28,
        seed=None,
        return_idx=False,
        return_ulbl_labels=False,
        crop_size=28,
        crop_ratio=1,
        download=True,
    ):
        """
        Initialise an DermaMNIST SSL dataset.  Contains a labelled, unlabelled , validation, evaluation dataset. All features are normalised.

        Elements from the datasets are return as dictionaries with keys
            "X": The original features as a tensor
            "weak": The weak augmentation applied to the features
            "strong": The strong augmentation applied to the features
            "y": The labels, if applicable
            "idx": The dataset index. This key is only returned if return_ulbl_labels=True

        Args:
            data_dir: The directory where the data is saved, or where it will be saved to if download=True
            lbls_per_class: The number of labelled observations to include per class
            ulbls_per_class: The number of unlabelled observations to include per class. By default all remaining unlabelled observations are used
            img_size: By default, use MedMNIST with 28x28. Specify 64, 128 or 224 for MedMNIST+ datasets.
            seed: The seed for randomly choosing the labelled instances
            crop_size: The length/width of crop size for resizing (square) during augmentations
            crop_ratio: The crop ratio used for padding when cropping during augmentations
            return_ulbl_labels: If true, the labels for the unlabelled data are included
            return_idx: If true, the indices are returned when accessing a dataset
            download: If true, the dataset is downloaded if it does not already exist
        """
        try:
            from medmnist import DermaMNIST
        except ImportError as e:
            raise ImportError(
                "This dataset requires `medmnist`. Install it with:\n"
                "    pip install sslpack[medmnist]"
            ) from e

        super().__init__(
            DermaMNIST,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            img_size,
            seed,
            return_idx,
            return_ulbl_labels,
            crop_size,
            crop_ratio,
            download,
        )


class BreastMnist(MedMnist):

    def __init__(
        self,
        data_dir,
        lbls_per_class,
        ulbls_per_class=None,
        img_size=28,
        seed=None,
        return_idx=False,
        return_ulbl_labels=False,
        crop_size=28,
        crop_ratio=1,
        download=True,
    ):
        """
        Initialise an BreastMNIST SSL dataset.  Contains a labelled, unlabelled , validation, evaluation dataset. All features are normalised.

        Elements from the datasets are return as dictionaries with keys
            "X": The original features as a tensor
            "weak": The weak augmentation applied to the features
            "strong": The strong augmentation applied to the features
            "y": The labels, if applicable
            "idx": The dataset index. This key is only returned if return_ulbl_labels=True

        Args:
            data_dir: The directory where the data is saved, or where it will be saved to if download=True
            lbls_per_class: The number of labelled observations to include per class
            ulbls_per_class: The number of unlabelled observations to include per class. By default all remaining unlabelled observations are used
            img_size: By default, use MedMNIST with 28x28. Specify 64, 128 or 224 for MedMNIST+ datasets.
            seed: The seed for randomly choosing the labelled instances
            crop_size: The length/width of crop size for resizing (square) during augmentations
            crop_ratio: The crop ratio used for padding when cropping during augmentations
            return_ulbl_labels: If true, the labels for the unlabelled data are included
            return_idx: If true, the indices are returned when accessing a dataset
            download: If true, the dataset is downloaded if it does not already exist
        """
        try:
            from medmnist import BreastMNIST
        except ImportError as e:
            raise ImportError(
                "This dataset requires `medmnist`. Install it with:\n"
                "    pip install sslpack[medmnist]"
            ) from e

        super().__init__(
            BreastMNIST,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            img_size,
            seed,
            return_idx,
            return_ulbl_labels,
            crop_size,
            crop_ratio,
            download,
        )

    def _permute(self, X):
        return X.permute(0, 1, 2)


class BloodMnist(MedMnist):

    def __init__(
        self,
        data_dir,
        lbls_per_class,
        ulbls_per_class=None,
        img_size=28,
        seed=None,
        return_idx=False,
        return_ulbl_labels=False,
        crop_size=28,
        crop_ratio=1,
        download=True,
    ):
        """
        Initialise an BloodMNIST SSL dataset.  Contains a labelled, unlabelled , validation, evaluation dataset. All features are normalised.

        Elements from the datasets are return as dictionaries with keys
            "X": The original features as a tensor
            "weak": The weak augmentation applied to the features
            "strong": The strong augmentation applied to the features
            "y": The labels, if applicable
            "idx": The dataset index. This key is only returned if return_ulbl_labels=True

        Args:
            data_dir: The directory where the data is saved, or where it will be saved to if download=True
            lbls_per_class: The number of labelled observations to include per class
            ulbls_per_class: The number of unlabelled observations to include per class. By default all remaining unlabelled observations are used
            img_size: By default, use MedMNIST with 28x28. Specify 64, 128 or 224 for MedMNIST+ datasets.
            seed: The seed for randomly choosing the labelled instances
            crop_size: The length/width of crop size for resizing (square) during augmentations
            crop_ratio: The crop ratio used for padding when cropping during augmentations
            return_ulbl_labels: If true, the labels for the unlabelled data are included
            return_idx: If true, the indices are returned when accessing a dataset
            download: If true, the dataset is downloaded if it does not already exist
        """
        try:
            from medmnist import BloodMNIST
        except ImportError as e:
            raise ImportError(
                "This dataset requires `medmnist`. Install it with:\n"
                "    pip install sslpack[medmnist]"
            ) from e

        super().__init__(
            BloodMNIST,
            data_dir,
            lbls_per_class,
            ulbls_per_class,
            img_size,
            seed,
            return_idx,
            return_ulbl_labels,
            crop_size,
            crop_ratio,
            download,
        )
