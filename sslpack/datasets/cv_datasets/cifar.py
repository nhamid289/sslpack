import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from PIL import Image

from sslpack.datasets import SSLDataset
from sslpack.utils.data import TransformDataset, BasicDataset, split_lb_ulb_balanced
from sslpack.utils.augmentation import RandAugment


class Cifar(SSLDataset):

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
        val_size=1/5,
    ):

        self.cifar = cifar
        self.return_ulbl_labels = return_ulbl_labels
        self.return_idx = return_idx
        self.val_size = val_size

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
        num_val = int(self.val_size*len(train))
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train), generator=generator)
        idx_val, idx_tr = idx[:num_val], idx[num_val:]

        X, y = train.data, torch.tensor(train.targets)
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_val, y_val = X[idx_val], y[idx_val]
        X_tr = [Image.fromarray(x) for x in X_tr]
        X_val = [Image.fromarray(x) for x in X_val]

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

        self.eval_dataset = BasicDataset(
            X=X_ts, y=y_ts, transform=self.transform, return_idx=self.return_idx
        )

        self.val_dataset = BasicDataset(
            X=X_val, y=y_val, transform=self.transform, return_idx=self.return_idx
        )

class Cifar10(Cifar):
    """
    A Cifar10 semi supervised learning dataset with transformations.

    """

    def __init__(
        self,
        data_dir,
        lbls_per_class=4,
        ulbls_per_class=None,
        return_ulbl_labels=False,
        return_idx=False,
        seed=None,
        crop_size=32,
        crop_ratio=1,
        val_size=1/6,
        download=False,
    ):
        """
        Initialise a CIFAR100 SSL dataset.  Contains a labelled, unlabelled and evaluation dataset. All features are normalised.

        Elements from the datasets are return as dictionaries with keys
            "X": The original features as a tensor
            "weak": The weak augmentation applied to the features
            "strong": The strong augmentation applied to the features
            "y": The labels, if applicable
            "idx": The dataset index, if enabled.

        Args:
            data_dir: The directory where the data is saved, or where it will be saved to if download=True
            lbls_per_class: The number of labelled observations to include per class
            ulbls_per_class: The number of unlabelled observations to include per class. By default all remaining unlabelled observations are used
            seed: The seed for randomly choosing the labelled instances
            crop_size: The length/width of crop size for resizing (square) during augmentations
            crop_ratio: The crop ratio used for padding when cropping during augmentations
            return_ulb_labels: If true, the labels for the unlabelled data are included
            return_idx: If true, the indices are returned when accessing a dataset
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
            return_ulbl_labels,
            return_idx,
            val_size
        )


class Cifar100(Cifar):
    """
    A Cifar100 semi supervised learning dataset with transformations
    """

    def __init__(
        self,
        data_dir,
        lbls_per_class=4,
        ulbls_per_class=None,
        return_ulbl_labels=False,
        return_idx=False,
        seed=None,
        crop_size=32,
        crop_ratio=1,
        val_size=1/5,
        download=False,
    ):
        """

        Initialise a CIFAR100 SSL dataset.  Contains a labelled, unlabelled and evaluation dataset. All features are normalised.

        Elements from the datasets are return as dictionaries with keys
            "X": The original features as a tensor
            "weak": The weak augmentation applied to the features
            "strong": The strong augmentation applied to the features
            "y": The labels, if applicable
            "idx": The dataset index, if enabled.

        Args:
            data_dir: The directory where the data is saved, or where it will be saved to if download=True
            lbls_per_class: The number of labelled observations to include per class
            ulbls_per_class: The number of unlabelled observations to include per class. By default all remaining unlabelled observations are used
            seed: The seed for randomly choosing the labelled instances
            crop_size: The length/width of crop size for resizing (square) during augmentations
            crop_ratio: The crop ratio used for padding when cropping during augmentations
            return_ulb_labels: If true, the labels for the unlabelled data are included
            return_idx: If true, the indices are returned when accessing a dataset
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
            return_idx,
            val_size
        )
