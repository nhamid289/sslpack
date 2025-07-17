import torch
from torch.nn.functional import cross_entropy

from wslearn.algorithms import Algorithm
from wslearn.utils.criterions import ce_consistency_loss

from .utils import _threshold_mask


class FixMatch(Algorithm):
    """An implementation of FixMatch (https://arxiv.org/pdf/2001.07685)

    By default the algorithm uses cross entropy loss for the supervised part,
    and cross entropy consistency loss for the unsupervised part.
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
        Initialise a fixmatch algorithm.

        Args:
            lambda_u: The weight of the unlabelled loss in the total loss.
            conf_threshold: the confidence threshold for pseudo-labels
            sup_loss_func: a function with signature f(pred, true) to compute
                the loss on the supervised batch. Default: cross entropy
            unsup_loss_func: a function with signature f(pred, true, mask) to
                compute the loss on the unsupervised batch: Default: masked cross entropy
        """
        super().__init__()

        self.lambda_u = lambda_u
        self.conf_threshold = conf_threshold
        self.max_pseudo_labels = max_pseudo_labels

        if sup_loss_func is None:
            # Default reduction is 'mean'
            self.sup_loss_func = cross_entropy
        else:
            self.sup_loss_func = sup_loss_func

        if unsup_loss_func is None:
            # Default reduction is 'mean'
            self.unsup_loss_func = ce_consistency_loss
        else:
            self.unsup_loss_func = unsup_loss_func

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

        x_lbl_weak = lbl_batch["weak"]
        x_ulbl_weak = ulbl_batch["weak"]
        x_ulbl_strong = ulbl_batch["strong"]

        confs, pseudo_labels, mask = _threshold_mask(
            model, x_ulbl_weak, self.conf_threshold, self.max_pseudo_labels
        )

        # with torch.no_grad():
        #     logits = model(x_ulbl_weak)
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

        x = torch.concat([x_lbl_weak, x_ulbl_strong])
        out = model(x)
        out_lbl_weak = out[: x_lbl_weak.size(0)]
        out_ulbl_strong = out[x_lbl_weak.size(0) :]

        sup_loss = self.sup_loss_func(out_lbl_weak, lbl_batch["y"])

        unsup_loss = self.unsup_loss_func(out_ulbl_strong, pseudo_labels, mask)

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
