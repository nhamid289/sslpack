from torch.utils.data import DataLoader
import torch

class WeaklySupervisedLoader(DataLoader):
    """
    An WeaklySupervisedLoader provides batches which contain both labelled and
    unlabelled data.
    """

    def __next__(self):
        """
        Return the next labelled and unlabelled batch.

        The return from __next__ should a 2-tuple containing a labelled
        and unlabelled batch
        """
        return NotImplementedError

class MinimumLoader(WeaklySupervisedLoader):
    """
    A dataloader which terminates after either the labelled or unlabelled
    has been exhausted.
    """

    def __init__(self, lbl_dataset, ulbl_dataset,
                 lbl_batch_size=1, ulbl_batch_size=1,
                 shuffle_lbl=True, shuffle_ulbl=True,
                 num_workers=0):

        self.lbl_dataset = lbl_dataset
        self.ulbl_dataset = ulbl_dataset

        self.lbl_loader = DataLoader(lbl_dataset, batch_size=lbl_batch_size,
                                     shuffle=shuffle_lbl, num_workers=num_workers)
        self.ulbl_loader = DataLoader(ulbl_dataset, batch_size=ulbl_batch_size,
                                      shuffle=shuffle_ulbl, num_workers=num_workers)

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

class CyclicLoader(WeaklySupervisedLoader):
    """
    A dataloader that continuously provides labelled and unlabelled batches.
    If either the labelled or unlabelled data is exhausted, it is reshuffled
    and the batches continue to be loaded.
    """

    def __init__(self, lbl_dataset, ulbl_dataset,
                 lbl_batch_size=1, ulbl_batch_size=1,
                 shuffle_lbl=True, shuffle_ulbl=True,
                 num_workers=0):

        self.lbl_dataset = lbl_dataset
        self.ulbl_dataset = ulbl_dataset

        self.lbl_loader = DataLoader(lbl_dataset, batch_size=lbl_batch_size,
                                     shuffle=shuffle_lbl, num_workers=num_workers)
        self.ulbl_loader = DataLoader(ulbl_dataset, batch_size=ulbl_batch_size,
                                      shuffle=shuffle_ulbl, num_workers=num_workers)

        self.lbl_iter = iter(self.lbl_loader)
        self.ulbl_iter = iter(self.ulbl_loader)

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
