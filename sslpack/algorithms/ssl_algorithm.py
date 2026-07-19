import torch.nn.functional as F
from torch import nn


class Algorithm(nn.Module):

    def __init__(self):
        super().__init__()

    def store_state(self, name: str, tensor):
        """Register a tensor as algorithm state that moves with .to(device)."""
        self.register_buffer(name, tensor)

    def warmup_forward(self, model, lbl_batch: dict, log_func=None):
        """
        Performs a supervised-only forward pass for use during a warmup phase.

        The default implementation computes cross-entropy loss on the labeled
        batch. Algorithms that need to accumulate statistics during warmup
        (e.g. Dash, which requires the average labeled loss to set its initial
        threshold) should override this method.

        Args:
            model (nn.Module):
                The classification model.
            lbl_batch (dict):
                A dictionary containing the labeled batch. Expects either a
                ``"weak"`` key (augmented features, preferred) or an ``"X"``
                key (raw features), and a ``"y"`` key for labels.
            log_func (Callable[[dict], None], optional):
                A function that accepts a dict with key ``"sup_loss"``.

        Returns:
            Tensor: the supervised cross-entropy loss.
        """
        x = lbl_batch.get("weak", lbl_batch.get("X"))
        if x is None:
            raise KeyError("lbl_batch must contain a 'weak' or 'X' key")
        logits = model(x)
        loss = F.cross_entropy(logits, lbl_batch["y"])
        if log_func is not None:
            log_func({"sup_loss": loss.item()})
        return loss

    def forward(self, model, lbl_batch: dict, ulbl_batch: dict, log_func=None):
        """
        forward specific to each algorithm
        """
        raise NotImplementedError
