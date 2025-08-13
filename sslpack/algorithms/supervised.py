import torch
from torch.nn.functional import cross_entropy as ce

from sslpack.algorithms import Algorithm

from sslpack.algorithms.utils import threshold_mask

class FullySupervised(Algorithm):
    """
    """

    def __init__(self, use_augment=False, use_weak=False, use_strong=False, loss_func=None):
        """
        Initialise a fixmatch algorithm.

        Args:
            lambda_u: The weight of the unlabelled loss in the total loss.
            conf_threshold: the confidence threshold for pseudo-labels
            concat: If false, a separate forward pass of the model for weakly augmented unlabelled examples is performed to generate pseudo-labels. If true, the labelled and unlabelled data are concatenated and a single forward pass is performed.
            max_pseudo_labels: The maximum number of pseudo-labels that can be used per batch. The highest confidence pseudo-labels are prioritised.
            sup_loss_func: a function with signature f(pred, true) to compute
                the loss on the supervised batch. Default: cross entropy
            unsup_loss_func: a function with signature f(pred, true, mask) to
                compute the loss on the unsupervised batch: Default: masked cross entropy
        """
        super().__init__()

        if use_augment is True:
            self.use_weak = True
            self.use_strong = True
        else:
            self.use_weak = use_weak
            self.use_strong = use_strong

        self.loss_func = ce if loss_func is None else loss_func

    def forward(self, model, lbl_batch, ulbl_batch, log_func=None):
        """
        Performs a forward pass of FixMatch

        Args:
            model: The predictor model
            lbl_batch: A dictionary containing the labelled batch. Expects keys
                "weak", "strong" for augmented features, "y" for labels
            ubl_batch: A dictionary with unlabelled data with keys. Expects keys
                "weak", "strong" for augmented features.
            log_func: A function which accepts a dictionary containing some
                training information
        """

        X = torch.concat([lbl_batch["X"], ulbl_batch["X"]])
        y = torch.concat([lbl_batch["y"], ulbl_batch["y"]])

        if self.use_weak is True:
            X = torch.concat([X, lbl_batch["weak"], ulbl_batch["weak"]])
            y = torch.concat([y, lbl_batch["y"], ulbl_batch["y"]])

        if self.use_strong is True:
            X = torch.concat([X, lbl_batch["strong"], ulbl_batch["strong"]])
            y = torch.concat([y, lbl_batch["y"], ulbl_batch["y"]])

        sup_loss = self.loss_func(model(X), y)
        if log_func is not None:
            log_func({
                "sup_loss": sup_loss.item(),
            })

        return sup_loss


class Supervised(Algorithm):
    """
    """

    def __init__(self, use_augment=False, use_weak=False, use_strong=False, loss_func=None):
        """
        Initialise a fixmatch algorithm.

        Args:
            lambda_u: The weight of the unlabelled loss in the total loss.
            conf_threshold: the confidence threshold for pseudo-labels
            concat: If false, a separate forward pass of the model for weakly augmented unlabelled examples is performed to generate pseudo-labels. If true, the labelled and unlabelled data are concatenated and a single forward pass is performed.
            max_pseudo_labels: The maximum number of pseudo-labels that can be used per batch. The highest confidence pseudo-labels are prioritised.
            sup_loss_func: a function with signature f(pred, true) to compute
                the loss on the supervised batch. Default: cross entropy
            unsup_loss_func: a function with signature f(pred, true, mask) to
                compute the loss on the unsupervised batch: Default: masked cross entropy
        """
        super().__init__()

        if use_augment is True:
            self.use_weak = True
            self.use_strong = True
        else:
            self.use_weak = use_weak
            self.use_strong = use_strong

        self.loss_func = ce if loss_func is None else loss_func

    def forward(self, model, lbl_batch, ulbl_batch, log_func=None):
        """
        Performs a forward pass of FixMatch

        Args:
            model: The predictor model
            lbl_batch: A dictionary containing the labelled batch. Expects keys
                "weak", "strong" for augmented features, "y" for labels
            ubl_batch: A dictionary with unlabelled data with keys. Expects keys
                "weak", "strong" for augmented features.
            log_func: A function which accepts a dictionary containing some
                training information
        """

        X = lbl_batch["X"]
        y = lbl_batch["y"]

        if self.use_weak is True:
            X = torch.concat([X, lbl_batch["weak"]])
            y = torch.concat([y, lbl_batch["y"]])

        if self.use_strong is True:
            X = torch.concat([X, lbl_batch["strong"]])
            y = torch.concat([y, lbl_batch["y"]])

        sup_loss = self.loss_func(model(X), y)
        if log_func is not None:
            log_func({
                "sup_loss": sup_loss.item(),
            })

        return sup_loss