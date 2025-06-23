import torch
from torch.nn.functional import cross_entropy

from wslearn.algorithms import Algorithm
from wslearn.utils.criterions import ce_consistency_loss

class FixMatch(Algorithm):
    """ An implementation of FixMatch (https://arxiv.org/pdf/2001.07685)

    By default the algorithm uses cross entropy loss for the supervised part,
    and cross entropy consistency loss for the unsupervised part.
    """

    def __init__(self, use_hard_label=False, lambda_u=1.0,
                 conf_threshold=0.95, sup_loss_func=None,
                 unsup_loss_func=None):
        """
        Initialise a fixmatch algorithm.

        Args:
            use_hard_label: true if using hard labelling for pseudo labels,
                otherwise soft labelling is used
            lambda_u: the weight of unlabelled loss in the total loss
            conf_threshold: the confidence threshold for pseudo-labels
            sup_loss_func: a function with signature f(pred, true) to compute
                the loss on the supervised batch
            unsup_loss_func: a function with signature f(pred, true, mask) to
                compute the loss on the unsupervised batch
        """
        super().__init__()

        self.use_hard_label = use_hard_label
        self.lambda_u = lambda_u
        self.conf_threshold = conf_threshold

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
            lbl_batch: A dictionary with labelled data using keys "X", "y"
            ubl_batch: A dictionary with unlabelled data using keys "X", "y"
            log_func: A function which accepts a dictionary containing some
                training information
        """
        out_lbl_weak = model(lbl_batch["weak"])
        out_ulbl_strong = model(ulbl_batch["strong"])

        with torch.no_grad():
            out_ulbl_weak = model(ulbl_batch["weak"])

        sup_loss = self.sup_loss_func(out_lbl_weak, lbl_batch["y"])

        probs_ulbl_w = torch.softmax(out_ulbl_weak.detach(), dim=-1)

        with torch.no_grad():
            max_probs, _ = torch.max(probs_ulbl_w, dim=-1)
            mask = max_probs.ge(self.conf_threshold).to(max_probs.dtype)

        if self.use_hard_label is True:
            pseudo_label = torch.argmax(probs_ulbl_w, dim=-1)
        else:
            pseudo_label = probs_ulbl_w

        unsup_loss = self.unsup_loss_func(out_ulbl_strong, pseudo_label, mask)

        total_loss = sup_loss + self.lambda_u * unsup_loss

        if log_func is not None:
            log_func({
                "sup_loss": sup_loss,
                "unsup_loss": unsup_loss,
                "total_loss": total_loss,
                "probs_ulbl": probs_ulbl_w,
                "mask": mask,
                "pseudo_label": pseudo_label
            })

        return total_loss