from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np


class BasicDataset(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        """
        Return the number of elements in the dataset
        """
        return(len(self.X))

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

        return out_dict


class TransformDataset(Dataset):
    """
    A class to store a dataset and apply any transformations to the data
    """
    def __init__(self, X, y=None, transform= None, weak_transform=None,
                 strong_transform=None, return_X_y=False):
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

        self.transform = transform
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        """
        Return the number of elements in the dataset
        """
        return(len(self.X))

    def __getitem__(self, index):
        """
        Returns an item from the dataset with any transformations applied

        Args:
            index: The index of the observation to return
        Returns:
            A 5-tuple (X, y, X_w, X_m, X_s) of the original data, label,
            medium and strong transformed data. If a weak transform is not
            specified, return the unmodified observation. If the data is
            unlabelled, or medium/strong is not specified, these all are None
        """
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

        return out_dict



