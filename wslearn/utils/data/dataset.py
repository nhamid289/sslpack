from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

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

        if isinstance(X, np.ndarray):
            X = Image.fromarray(X)

        X= transforms.ToTensor()(X)

        return X, y


class TransformDataset(Dataset):
    """
    A class to store a dataset and apply any transformations to the data
    """
    def __init__(self, X, y=None, weak_transform=None, medium_transform=None,
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

        self.weak_transform = weak_transform
        self.medium_transform = medium_transform
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
        if isinstance(X, np.ndarray):
            X = Image.fromarray(X)
        X_orig = transforms.ToTensor()(X)

        if self.return_X_y is True:
            X_w = self.weak_transform(X) if self.weak_transform is not None else X_orig
            X_m = self.medium_transform(X) if self.medium_transform is not None else X_orig
            X_s = self.strong_transform(X) if self.strong_transform is not None else X_orig

            return X_orig, X_w, X_m, X_s, y

        out_dict = {"X": X_orig}
        if y is not None:
            out_dict["y"] = y
        if self.weak_transform is not None:
            out_dict["weak"] = self.weak_transform(X)
        if self.medium_transform is not None:
            out_dict["medium"] = self.medium_transform(X)
        if self.strong_transform is not None:
            out_dict["strong"] = self.strong_transform(X)

        return out_dict



