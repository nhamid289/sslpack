from torch import nn


class Algorithm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, model, lbl_batch: dict, ulbl_batch: dict, log_func=None):
        """
        forward specific to each algorithm
        """
        raise NotImplementedError
