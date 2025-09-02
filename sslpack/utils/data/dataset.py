from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np


class BasicDataset(Dataset):
    """
    An sslpack BasicDataset to store features and (optionally) labels.

    Args:
        X:
            The features
        y (optional):
            The labels (if applicable)
        transform (optional):
            The transform to apply to the features X when accessed. If unspecified, no transform is applied.
        return_idx (bool):
            If true, the index of observations is available with key "idx". Defaults to False.

    Elements are returned as a dictionary with keys
    - "X": The features (after applying the transformation)
    - "y": The label (if applicable)
    - "idx": The data index (if return_idx=True)

    """
    def __init__(self, X, y=None, transform=None, return_idx=False):
        self.X = X
        self.y = y
        self.return_idx = return_idx
        self.transform = transform

    def __len__(self):
        """
        Return the number of elements in the dataset
        """
        return len(self.X)

    def __getitem__(self, index):
        """
        Returns an item from the dataset
        """
        X = self.X[index]
        y = self.y[index] if self.y is not None else None

        if self.transform is not None:
            out_dict = {"X": self.transform(X)}
        else:
            out_dict = {"X": X}

        if y is not None:
            out_dict["y"] = y

        if self.return_idx is True:
            out_dict["idx"] = index

        return out_dict


class TransformDataset(Dataset):
    """
    An sslpack TransformDataset to store features and (optionally) labels.
    Compared to BasicDataset, TransformDataset is designed for augmentation anchoring methods, with support for two augmentations.

    Args:
        X:
            The features
        y (optional):
            The labels (if applicable)
        transform (optional):
            The basic transform to apply to the features. Access with "X". If unspecified, no transform is applied.
        weak_transform (optional):
            The weak transform to apply to the features. Access with "weak". Defaults to basic transform if unspecified.
        transform (optional):
            The strong transform to apply to the features. Access with "strong". Defaults to basic transform if unspecified.
        return_X_y (bool):
            If true, rather than a dictionary, outputs are returned in a tuple (X, X_weak, X_strong, y). Defaults to False.
        return_idx (bool):
            If true, the index of observations is available with key "idx". Defaults to False.

    Elements are returned as a dictionary with keys
    - "X": The features (after applying the transformation)
    - "y": The label (if applicable)
    - "idx": The data index (if return_idx=True)

    """

    def __init__(
        self,
        X,
        y=None,
        transform=None,
        weak_transform=None,
        strong_transform=None,
        return_X_y=False,
        return_idx=False
    ):
        """
        Initialise an WSL dataset. This can be either a labelled or unlabelled
        dataset.

        Using __getitem__ returns a dictionary with the features, any
        transformations, and the label if these are present using the following
        keys: "X", "y", "weak", "medium", "strong"

        Args:
            X: the features of the data
            y: the class labels
            weak_transform: a basic transformation that is always applied
            medium_transform: a transformation that may be applied
            strong_transform: a transformation that may be applied
            return_X_y: if True, __getitem__ returns all outputs
            as a tuple instead of a dictionary
        """

        super().__init__()

        self.X = X
        self.y = y
        self.return_X_y = return_X_y
        self.return_idx = return_idx

        self.transform = transform
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        """
        Return the number of elements in the dataset
        """
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index] if self.y is not None else None

        if self.return_X_y is True:
            X_t = self.transform(X) if self.transform is not None else X
            X_w = self.weak_transform(X) if self.weak_transform is not None else X_t
            X_s = self.strong_transform(X) if self.strong_transform is not None else X_t
            return X_t, X_w, X_s, y

        out_dict = {}
        if y is not None:
            out_dict["y"] = y
        if self.weak_transform is not None:
            out_dict["weak"] = self.weak_transform(X)
        if self.strong_transform is not None:
            out_dict["strong"] = self.strong_transform(X)
        if self.transform is not None:
            out_dict["X"] = self.transform(X)
        else:
            out_dict["X"] = X
        if self.return_idx:
            out_dict["idx"] = index

        return out_dict
