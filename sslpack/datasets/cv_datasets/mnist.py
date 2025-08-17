from sslpack.datasets import Dataset
from torchvision.datasets import MNIST as MN
import torch
from torch.utils.data import random_split
from torchvision import transforms
from sslpack.utils.data import TransformDataset, BasicDataset
from sslpack.utils.data import split_lb_ulb_balanced
from sslpack.utils.augmentation import RandAugment


class Mnist(Dataset):

    def __init__(
        self,
        data_dir,
        lbls_per_class,
        ulbls_per_class=None,
        seed=None,
        return_idx=False,
        return_ulbl_labels=False,
        crop_size=28,
        crop_ratio=1,
        val_size=1/6,
        download=False,
    ):
        """
        Initialise an MNIST SSL dataset.  Contains a labelled, unlabelled and evaluation dataset. All features are normalised.

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
            seed: The seed for randomly choosing the labelled instances
            crop_size: The length/width of crop size for resizing (square) during augmentations
            crop_ratio: The crop ratio used for padding when cropping during augmentations
            return_ulbl_labels: If true, the labels for the unlabelled data are included
            return_idx: If true, the indices are returned when accessing a dataset
            val_size: The proportion of training data to use as validation set
            download: If true, the dataset is downloaded if it does not already exist
        """

        mnist_tr = MN(root=data_dir, train=True, download=download)
        mnist_ts = MN(root=data_dir, train=False, download=download)

        num_val = int(val_size*len(mnist_tr))
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(mnist_tr), generator=torch.Generator().manual_seed(seed))
        idx_val, idx_tr = idx[:num_val], idx[num_val:]

        X, y = mnist_tr.data.float() / 255, mnist_tr.targets
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_val, y_val = X[idx_val], y[idx_val]
        X_ts, y_ts = mnist_ts.data.float() / 255, mnist_ts.targets

        X_tr, X_ts, X_val = X_tr.unsqueeze(1), X_ts.unsqueeze(1), X_val.unsqueeze(1)

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

        X_ts, y_ts = X_ts.float(), y_ts.float()

        self.eval_dataset = BasicDataset(
            X_ts, y_ts, transform=self.transform, return_idx=return_idx
        )

        self.val_dataset = BasicDataset(
            X_val, y_val, transform=self.transform, return_idx=return_idx
        )
