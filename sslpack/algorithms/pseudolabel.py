from sslpack.algorithms import Algorithm
from sslpack.utils.criterions import ce_consistency_loss as cel

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy as ce

from typing import Optional, Callable

from sslpack.algorithms.utils import threshold_mask, DistributionAlignment

class PseudoLabel(Algorithm):
    """
    An implementation of PseudoLabel algorithm for weakly supervised training.

    Args:
        lambda_u (float, optional):
            The weight of the unlabelled loss in the total loss. Expects non-negative real >= 0. Defaults to 1.
        conf_threshold (float, optional):
            The default confidence threshold for pseudo-labels. Expects a float in [0, 1]. Defaults to 0.95
        concat (bool, optional):
            If True, the labelled and unlabelled batches are concatenated, and a single forward pass of the model is performed.
            If False, a separate forward pass is performed for each of the labelled and unlabelled batches.
            Defaults to True.
        use_dist_align (bool, optional):
            If True, distribution alignment is performed on the weakly-augmented unlabelled output prior to pseudo-labelling. Defaults to False.
        dist_align (Callable[[Tensor, Tensor], Tensor], optional):
            The distribution alignment function used. Expects a callable with arguments f(probs_ulbl, probs_lbl) which
            are normalised vectors with dimension matching num_classes. By default, the sslpack DistributionAlignment implementation is used.
        max_pseudo_labels (int, optional):
            The maximum number of pseudo-labels to use in each unlabelled batch. If None, all pseudo-labels above the threshold are accepted.
            Higher confidence labels are prioritised. Defaults to None.
        sup_loss_func (Callable[[Tensor, Tensor], Tensor], optional):
            a function with signature f(pred, true) to compute
            the loss on the supervised batch. Defaults to torch cross_entropy
        unsup_loss_func (Callable[[Tensor, Tensor, Tensor], Tensor], optional):
            a function with signature f(pred, true, mask) compute the loss on the unsupervised batch for only unmasked examples.
            Defaults to sslpack's masked cross entropy

    """

    def __init__(self,
                 lambda_u:float=1,
                 conf_threshold:float=0.95,
                 concat:bool=True,
                 use_dist_align:bool=False,
                 dist_align:Optional[Callable[[Tensor, Tensor], Tensor]]=None,
                 max_pseudo_labels:Optional[int]=None,
                 sup_loss_func:Optional[Callable[[Tensor, Tensor], Tensor]]=None,
                 unsup_loss_func:Optional[Callable[[Tensor, Tensor, Tensor], Tensor]]=None
                 ):
        """
        Initialise a PseudoLabel algorithm
        """

        self.lambda_u = lambda_u
        self.conf_threshold = conf_threshold
        self.max_pseudo_labels = max_pseudo_labels
        self.concat = concat
        self.use_dist_align = use_dist_align
        if dist_align is None:
            self.dist_align = DistributionAlignment()
        else:
            self.dist_align = dist_align

        self.sup_loss_func = ce if sup_loss_func is None else sup_loss_func
        self.unsup_loss_func = cel if unsup_loss_func is None else unsup_loss_func

    def forward(self,
                model:nn.Module,
                lbl_batch:dict,
                ulbl_batch:dict,
                log_func:Optional[Callable[[dict], None]]=None):
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

        if self.concat is True:
            x = torch.concat([x_lbl, x_ulbl])
            out = model(x)
            out_lbl = out[:x_lbl.size(0)]
            out_ulbl = out[x_lbl.size(0):]
        else:
            out_lbl = model(x_lbl)
            with torch.no_grad():
                out_ulbl = model(x_ulbl)

        probs_ulbl = torch.softmax(out_ulbl, dim=1)
        probs_lbl = torch.softmax(out_lbl, dim=1)

        if self.use_dist_align is True:
            probs_ulbl = self.dist_align(probs_ulbl, probs_lbl)

        # generate pseudo-labels
        confs, pseudo_labels, mask = threshold_mask(probs_ulbl,
                                                    self.conf_threshold,
                                                    self.max_pseudo_labels)

        sup_loss = self.sup_loss_func(out_lbl, lbl_batch["y"])
        unsup_loss = self.unsup_loss_func(out_ulbl, pseudo_labels, mask)
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
