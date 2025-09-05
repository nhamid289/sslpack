from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset

from typing import Tuple, Union, Optional


class SSLDataLoader(DataLoader):
    """
    An SSLDataLoader  provides batches which contain both labelled and
    unlabelled data.
    """
    def __init__(
        self,
        lbl_dataset:Dataset,
        ulbl_dataset:Dataset,
        lbl_batch_size:int=1,
        ulbl_batch_size:int=1,
        u_ratio:Optional[float]=None,
        shuffle_lbl=True,
        shuffle_ulbl=True,
        num_workers=0,
    ):
        self.lbl_dataset = lbl_dataset
        self.ulbl_dataset = ulbl_dataset
        self.lbl_batch_size = lbl_batch_size
        if u_ratio is not None:
            self.ulbl_batch_size = u_ratio * lbl_batch_size
        else:
            self.ulbl_batch_size = ulbl_batch_size
        self.shuffle_lbl= shuffle_lbl
        self.shuffle_ulbl = shuffle_ulbl
        if isinstance(num_workers, tuple):
            self.lbl_workers, self.ulbl_workers = num_workers
        else:
            self.lbl_workers, self.ulbl_workers = num_workers, num_workers

        self.lbl_loader = DataLoader(
            self.lbl_dataset,
            batch_size=self.lbl_batch_size,
            shuffle=self.shuffle_lbl,
            num_workers=self.lbl_workers,
        )
        self.ulbl_loader = DataLoader(
            self.ulbl_dataset,
            batch_size=self.ulbl_batch_size,
            shuffle=self.shuffle_ulbl,
            num_workers=self.ulbl_workers,
        )

        self.lbl_iter = iter(self.lbl_loader)
        self.ulbl_iter = iter(self.ulbl_loader)

    def __len__(self):
        """
        Length
        """
        return NotImplementedError

    def __iter__(self):
        """
        Iterate
        """
        return NotImplementedError

    def __next__(self):
        """
        Return the next labelled and unlabelled batch.
        """
        return NotImplementedError


class LabelledEpochLoader(SSLDataLoader):
    """
    An SSL dataloader which terminates after the labelled data have been exhausted.

    Args:
        lbl_dataset (torch.utils.data.Dataset):
            The labelled dataset
        ulbl_dataset (torch.utils.data.Dataset):
            The unlabelled dataset
        lbl_batch_size (int):
            The labelled batch size. Expects an integer > 0. Defaults to 1.
        ulbl_batch_size (int):
            The unlabelled batch size. Expects an integer > 0. Defaults to 1 unless u_ratio is specified.
        u_ratio (float, optional):
            The ratio of unlabelled batch size to labelled batch size. If specified, ulbl_batch_size is ignored.
        shuffle_lbl (bool):
            If true, the labelled data are shuffled prior to dividing into batches.
        shuffle_ulbl (bool):
            If true, the unlabelled data are shuffled prior to dividing into batches.
        num_workers (int or Tuple[int, int], optional):
            The number of workers allocated to the labelled and unlabelled data.
            If a singleton, both datasets use the same number of workers.
            If a tuple is specified, the first element corresponds to the labelled workers,
            and the second is for the unlabelled workers

    """
    def __init__(
        self,
        lbl_dataset:Dataset,
        ulbl_dataset:Dataset,
        lbl_batch_size:int=1,
        ulbl_batch_size:int=1,
        u_ratio:Optional[float]=None,
        shuffle_lbl=True,
        shuffle_ulbl=True,
        num_workers=0,
    ):
        super().__init__(lbl_dataset,
                         ulbl_dataset,
                         lbl_batch_size,
                         ulbl_batch_size,
                         u_ratio,
                         shuffle_lbl,
                         shuffle_ulbl,
                         num_workers)

    def __iter__(self):
        self.lbl_iter = iter(self.lbl_loader)
        return self

    def __next__(self):
        lbl_batch = next(self.lbl_iter)
        ulbl_batch = next(self.ulbl_iter)
        return lbl_batch, ulbl_batch

    def __len__(self):
        return min(len(self.lbl_loader), len(self.ulbl_loader))


class UnlabelledEpochLoader(SSLDataLoader):
    """
    An SSL dataloader which terminates after the labelled data have been exhausted.

    Args:
        lbl_dataset (torch.utils.data.Dataset):
            The labelled dataset
        ulbl_dataset (torch.utils.data.Dataset):
            The unlabelled dataset
        lbl_batch_size (int):
            The labelled batch size. Expects an integer > 0. Defaults to 1.
        ulbl_batch_size (int):
            The unlabelled batch size. Expects an integer > 0. Defaults to 1 unless u_ratio is specified.
        u_ratio (float, optional):
            The ratio of unlabelled batch size to labelled batch size. If specified, ulbl_batch_size is ignored.
        shuffle_lbl (bool):
            If true, the labelled data are shuffled prior to dividing into batches.
        shuffle_ulbl (bool):
            If true, the unlabelled data are shuffled prior to dividing into batches.
        num_workers (int or Tuple[int, int], optional):
            The number of workers allocated to the labelled and unlabelled data.
            If a singleton, both datasets use the same number of workers.
            If a tuple is specified, the first element corresponds to the labelled workers,
            and the second is for the unlabelled workers
    """
    def __init__(
        self,
        lbl_dataset:Dataset,
        ulbl_dataset:Dataset,
        lbl_batch_size:int=1,
        ulbl_batch_size:int=1,
        u_ratio:Optional[float]=None,
        shuffle_lbl=True,
        shuffle_ulbl=True,
        num_workers=0,
    ):
        super().__init__(lbl_dataset,
                         ulbl_dataset,
                         lbl_batch_size,
                         ulbl_batch_size,
                         u_ratio,
                         shuffle_lbl,
                         shuffle_ulbl,
                         num_workers)

    def __iter__(self):
        self.ulbl_iter = iter(self.ulbl_loader)
        return self

    def __next__(self):
        ulbl_batch = next(self.ulbl_iter)
        try:
            lbl_batch = next(self.lbl_iter)
        except StopIteration:
            self.lbl_iter = iter(self.lbl_loader)
            lbl_batch = next(self.lbl_iter)
        return lbl_batch, ulbl_batch

    def __len__(self):
        return len(self.ulbl_loader)


class MinimumLoader(SSLDataLoader):
    """
    An SSL dataloader which terminates after either the labelled or unlabelled data have been exhausted.

    Args:
        lbl_dataset (torch.utils.data.Dataset):
            The labelled dataset
        ulbl_dataset (torch.utils.data.Dataset):
            The unlabelled dataset
        lbl_batch_size (int):
            The labelled batch size. Expects an integer > 0. Defaults to 1.
        ulbl_batch_size (int):
            The unlabelled batch size. Expects an integer > 0. Defaults to 1 unless u_ratio is specified.
        u_ratio (float, optional):
            The ratio of unlabelled batch size to labelled batch size. If specified, ulbl_batch_size is ignored.
        shuffle_lbl (bool):
            If true, the labelled data are shuffled prior to dividing into batches.
        shuffle_ulbl (bool):
            If true, the unlabelled data are shuffled prior to dividing into batches.
        num_workers (int or Tuple[int, int], optional):
            The number of workers allocated to the labelled and unlabelled data.
            If a singleton, both datasets use the same number of workers.
            If a tuple is specified, the first element corresponds to the labelled workers,
            and the second is for the unlabelled workers
    """
    def __init__(
        self,
        lbl_dataset:Dataset,
        ulbl_dataset:Dataset,
        lbl_batch_size:int=1,
        ulbl_batch_size:int=1,
        u_ratio:Optional[float]=None,
        shuffle_lbl=True,
        shuffle_ulbl=True,
        num_workers=0,
    ):
        super().__init__(lbl_dataset,
                         ulbl_dataset,
                         lbl_batch_size,
                         ulbl_batch_size,
                         u_ratio,
                         shuffle_lbl,
                         shuffle_ulbl,
                         num_workers)

    def __iter__(self):
        self.lbl_iter = iter(self.lbl_loader)
        self.ulbl_iter = iter(self.ulbl_loader)
        return self

    def __next__(self):
        lbl_batch = next(self.lbl_iter)
        ulbl_batch = next(self.ulbl_iter)
        return lbl_batch, ulbl_batch

    def __len__(self):
        return min(len(self.lbl_loader), len(self.ulbl_loader))


class CyclicLoader(SSLDataLoader):
    """
    An SSL dataloader which never terminates, and instead refreshes the
    labelled and unlabelled loaders whenever they are exhausted. Make sure to terminate this dataloader
    manually in code.

    Args:
        lbl_dataset (torch.utils.data.Dataset):
            The labelled dataset
        ulbl_dataset (torch.utils.data.Dataset):
            The unlabelled dataset
        lbl_batch_size (int):
            The labelled batch size. Expects an integer > 0. Defaults to 1.
        ulbl_batch_size (int):
            The unlabelled batch size. Expects an integer > 0. Defaults to 1 unless u_ratio is specified.
        u_ratio (float, optional):
            The ratio of unlabelled batch size to labelled batch size. If specified, ulbl_batch_size is ignored.
        shuffle_lbl (bool):
            If true, the labelled data are shuffled prior to dividing into batches.
        shuffle_ulbl (bool):
            If true, the unlabelled data are shuffled prior to dividing into batches.
        num_workers (int or Tuple[int, int], optional):
            The number of workers allocated to the labelled and unlabelled data.
            If a singleton, both datasets use the same number of workers.
            If a tuple is specified, the first element corresponds to the labelled workers,
            and the second is for the unlabelled workers
    """
    def __init__(
        self,
        lbl_dataset:Dataset,
        ulbl_dataset:Dataset,
        lbl_batch_size:int=1,
        ulbl_batch_size:int=1,
        u_ratio:Optional[float]=None,
        shuffle_lbl=True,
        shuffle_ulbl=True,
        num_workers=0,
    ):
        super().__init__(lbl_dataset,
                         ulbl_dataset,
                         lbl_batch_size,
                         ulbl_batch_size,
                         u_ratio,
                         shuffle_lbl,
                         shuffle_ulbl,
                         num_workers)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            lbl_batch = next(self.lbl_iter)
        except StopIteration:
            self.lbl_iter = iter(self.lbl_loader)
            lbl_batch = next(self.lbl_iter)

        try:
            ulbl_batch = next(self.ulbl_iter)
        except StopIteration:
            self.ulbl_iter = iter(self.ulbl_loader)
            ulbl_batch = next(self.ulbl_iter)

        return lbl_batch, ulbl_batch
