from wslearn.algorithms import Algorithm
from wslearn.utils.criterions import ce_consistency_loss

import torch
import torch.nn.functional as F

from .utils import _threshold_mask


class PseudoLabel(Algorithm):
    """
    An implementation of PseudoLabel algorithm for weakly supervised training.
    """

    def __init__(
        self,
        lambda_u=1,
        conf_threshold=0.95,
        max_pseudo_labels=None,
        sup_loss_func=None,
        unsup_loss_func=None,
    ):
        """
        Initialise a PseudoLabel algorithm

        Args:
            lambda_u: The weight of the unlabelled loss in the total loss.
            conf_threshold: The minimum confidence level needed for selecting
                pseudo labels
            max_pseudo_labels: The maximum number of pseudo labels to use
                when computing unsupervised loss
            sup_loss_func: a function with signature f(pred, true) to compute
                the loss on the supervised batch. Default: Cross entropy
            unsup_loss_func: a function with signature f(pred, true, mask) to
                compute the loss on the unsupervised batch. Default: Masked cross entropy
        """

        self.lambda_u = lambda_u
        self.conf_threshold = conf_threshold
        self.max_pseudo_labels = max_pseudo_labels

        if sup_loss_func is None:
            # Default reduction is 'mean'
            self.sup_loss_func = F.cross_entropy
        else:
            self.sup_loss_func = sup_loss_func

        if unsup_loss_func is None:
            # Default reduction is 'mean'
            self.unsup_loss_func = ce_consistency_loss
        else:
            self.unsup_loss_func = unsup_loss_func

    def forward(self, model, lbl_batch, ulbl_batch, log_func=None):
        """
        Perform a forward pass of PseudoLabel.

        Args:
            model: The predictor model
            lbl_batch: A dictionary with labelled data using keys "X", "y"
            ubl_batch: A dictionary with unlabelled data using keys "X", "y"
            log_func: A function which accepts a dictionary containing some
                training information
        """
        x_lbl = lbl_batch["X"]
        x_ulbl = ulbl_batch["X"]

        # generate pseudo-labels

        confs, pseudo_labels, mask = _threshold_mask(
            model, x_ulbl, self.conf_threshold, self.max_pseudo_labels
        )

        # confs, pseudo_labels, mask = _mask(model, conf_threshol, max_pseudo_labels)
        # with torch.no_grad():
        #     logits = model(x_ulbl)
        #     probs = torch.softmax(logits, dim=1)
        #     confidences, pseudo_labels = torch.max(probs, dim=1)
        #     mask = confidences.ge(self.conf_threshold)
        #     if self.max_pseudo_labels is not None:
        #         _, indices = torch.topk(confidences,
        #                                 min(self.max_pseudo_labels,
        #                                     len(confidences)))
        #         keep = torch.zeros_like(mask, dtype=torch.bool)
        #         keep[indices] = True
        #         mask &= keep

        x = torch.concat([x_lbl, x_ulbl])
        out = model(x)
        out_lbl = out[: x_lbl.size(0)]
        out_ulbl = out[x_lbl.size(0) :]

        sup_loss = self.sup_loss_func(out_lbl, lbl_batch["y"])

        unsup_loss = self.unsup_loss_func(out_ulbl, pseudo_labels, mask)

        total_loss = sup_loss + self.lambda_u * unsup_loss

        if log_func is not None:
            log_func(
                {
                    "sup_loss": sup_loss.item(),
                    "unsup_loss": unsup_loss.item(),
                    "total_loss": total_loss.item(),
                    "confidences": confs.detach().cpu(),
                    "pseudo_labels": pseudo_labels.detach().cpu(),
                    "mask": mask.detach().cpu(),
                }
            )

        return total_loss
