from torch import nn


class Algorithm(nn.Module):

    def __init__(self):
        super().__init__()

    def store_state(self, name: str, tensor):
        """Register a tensor as algorithm state that moves with .to(device)."""
        self.register_buffer(name, tensor)

    def forward(self, model, lbl_batch: dict, ulbl_batch: dict, log_func=None):
        """
        forward specific to each algorithm
        """
        raise NotImplementedError
