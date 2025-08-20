import torch
from torch.nn.functional import cross_entropy

from sslpack.algorithms import Algorithm
from sslpack.utils.criterions import ce_consistency_loss

from sslpack.algorithms.utils import threshold_mask

class FixMatch(Algorithm):
    """ An implementation of FixMatch (https://arxiv.org/pdf/2001.07685)

    By default the algorithm uses cross entropy loss for the supervised part,
    and cross entropy consistency loss for the unsupervised part.
    """

    def __init__(self, lambda_u=1, conf_threshold=0.95, concat=True,
                 dist_align = None, max_pseudo_labels=None, sup_loss_func=None,
                 unsup_loss_func=None):
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

        self.lambda_u = lambda_u
        self.conf_threshold = conf_threshold
        self.max_pseudo_labels = max_pseudo_labels
        self.concat = concat
        self.dist_align = dist_align

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

        lbl_size = lbl_batch["weak"].size(0)
        ulbl_size = ulbl_batch["weak"].size(0)

        if self.concat is True:
            x = torch.concat([lbl_batch["weak"], ulbl_batch["weak"], ulbl_batch["strong"]])
            out = model(x)
            out_lbl_weak = out[:lbl_size] # unconcat the outputs
            out_ulbl_weak = out[lbl_size: lbl_size + ulbl_size]
            out_ulbl_strong = out[lbl_size + ulbl_size:]
        else:
            x = torch.concat([lbl_batch["weak"], ulbl_batch["strong"]])
            out = model(x)
            out_lbl_weak = out[:lbl_size]
            out_ulbl_strong = out[lbl_size:]
            out_ulbl_weak = model(ulbl_batch["weak"])

        probs_ubl_weak = torch.softmax(out_ulbl_weak, dim=1)

        if self.dist_align is not None:
            probs_ubl_weak = self.dist_align(probs_ubl_weak,out_lbl_weak.softmax(dim=1))
            # probs_ubl_weak = self.dist_align(probs_ubl_weak)

        # pseudo-labels are generated under no_grad()
        confs, pseudo_labels, mask = threshold_mask(probs_ubl_weak,
                                                     self.conf_threshold,
                                                     self.max_pseudo_labels)

        sup_loss = self.sup_loss_func(out_lbl_weak, lbl_batch["y"])
        unsup_loss = self.unsup_loss_func(out_ulbl_strong, pseudo_labels, mask)
        total_loss = sup_loss + self.lambda_u * unsup_loss

        if log_func is not None:
            log_func({
                "sup_loss": sup_loss.item(),
                "unsup_loss": unsup_loss.item(),
                "total_loss": total_loss.item(),
                "confidences": confs.detach().cpu(),
                "pseudo_labels": pseudo_labels.detach().cpu(),
                "mask": mask.detach().cpu()
            })

        return total_loss