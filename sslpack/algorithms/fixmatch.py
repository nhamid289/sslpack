import torch
from torch.nn.functional import cross_entropy as ce

from sslpack.algorithms import Algorithm
from sslpack.utils.criterions import ce_consistency_loss as cel

from sslpack.algorithms.utils import threshold_mask, DistributionAlignment

class FixMatch(Algorithm):
    """ An implementation of FixMatch (https://arxiv.org/pdf/2001.07685)

    By default the algorithm uses cross entropy loss for the supervised part,
    and cross entropy consistency loss for the unsupervised part.
    """

    def __init__(self, lambda_u=1, conf_threshold=0.95, concat=True, use_dist_align=False,
                 dist_align=None, max_pseudo_labels=None, sup_loss_func=None, unsup_loss_func=None):
        """
        Initialise a fixmatch algorithm.

        Args:
            lambda_u: The weight of the unlabelled loss in the total loss.
            conf_threshold: the confidence threshold for pseudo-labels
            concat: If false, a separate forward pass is performed for each labelled/unlabelled batches.
                If true, the labelled and unlabelled data are concatenated and a single forward pass is performed.
            use_dist_align: If true, distribution alignment is used
            dist_align: The distribution alignment object to use.
                Expects a callable with signature dist_align(probs_ulb, probs_lb)
            max_pseudo_labels: The maximum number of pseudo-labels that can be used per batch.
                The highest confidence pseudo-labels are prioritised.
            sup_loss_func: The method to comppute the loss on the labelled batch.
                Expects a callable with signature f(pred, true) to compute
                the loss on the supervised batch. Default: cross entropy
            unsup_loss_func: a function with signature f(pred, true, mask) to
                compute the loss on the unsupervised batch: Default: masked cross entropy
        """
        super().__init__()

        self.lambda_u = lambda_u
        self.conf_threshold = conf_threshold
        self.max_pseudo_labels = max_pseudo_labels
        self.concat = concat
        self.use_dist_align = use_dist_align
        if dist_align is None:
            self.dist_align = DistributionAlignment()
        else:
            self.dist_align = dist_align

        # Default reduction for loss functions is 'mean'
        self.sup_loss_func = ce if sup_loss_func is None else sup_loss_func
        self.unsup_loss_func = cel if unsup_loss_func is None else unsup_loss_func

    def _model_outputs(self, model, lbl_batch, ulbl_batch):

        lbl_size = lbl_batch["weak"].size(0)
        ulbl_size = ulbl_batch["weak"].size(0)

        if self.concat is True:
            x = torch.concat([lbl_batch["weak"], ulbl_batch["weak"], ulbl_batch["strong"]])
            o = model(x)
            o_lbl_w = o[:lbl_size] # unconcat the outputs
            o_ulbl_w = o[lbl_size: lbl_size + ulbl_size]
            o_ulbl_s = o[lbl_size + ulbl_size:]
        else:
            o_lbl_w = model(lbl_batch["weak"])
            o_ulbl_s = model(ulbl_batch["strong"])
            with torch.no_grad():
                o_ulbl_w = model(ulbl_batch["weak"])

        return o_lbl_w, o_ulbl_w, o_ulbl_s


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

        # model outputs on labelled/unlabelled examples with augmentations
        o_lbl_w, o_ulbl_w, o_ulbl_s = self._model_outputs(model, lbl_batch, ulbl_batch)

        probs_lbl_w, probs_ulbl_w = o_lbl_w.softmax(dim=1), o_ulbl_w.softmax(dim=1)

        if self.use_dist_align is True:
            probs_ulbl_w = self.dist_align(probs_ulbl_w, probs_lbl_w)

        # pseudo-labels are generated under no_grad()
        confs, pseudo_labels, mask = threshold_mask(probs_ulbl_w,
                                                     self.conf_threshold,
                                                     self.max_pseudo_labels)

        sup_loss = self.sup_loss_func(o_lbl_w, lbl_batch["y"])
        unsup_loss = self.unsup_loss_func(o_ulbl_s, pseudo_labels, mask)
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