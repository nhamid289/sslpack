import torch
from torch.nn.functional import cross_entropy as ce

from sslpack.algorithms import Algorithm
from sslpack.algorithms.utils import DistributionAlignment
from sslpack.utils.criterions import ce_consistency_loss as cel

from torch import Tensor, device, nn
from typing import Optional, Callable, Union

class FlexMatch(Algorithm):
    """ An implementation of FlexMatch (http://arxiv.org/abs/2110.08263)

    FlexMatch uses pseudo-labelling and augmentation anchoring like FixMatch, but adaptively selects
    the confidence threshold on a class-wise basis.

    Args:
        num_classes (int):
            The number of classes. Expects positive integer > 0.
        num_ulbl (int):
            The number of unlabelled training examples. Expects positive integer > 0
        lambda_u (float, optional):
            The weight of the unlabelled loss in the total loss. Expects non-negative real >= 0. Defaults to 1.
        conf_threshold (float, optional):
            The default confidence threshold for pseudo-labels. Expects a float in [0, 1]. Defaults to 0.95
        use_warmup (bool, optional):
            If true, use threshold warmup when most data remains unseen. Default False.
        concat (bool, optional):
            If True, the labelled and unlabelled batches are concatenated, and a single forward pass of the model is performed.
            If False, a separate forward pass is performed for each of the labelled and unlabelled batches.
            Defaults to True.
        use_dist_align (bool, optional):
            If True, distribution alignment is performed on the weakly-augmented unlabelled output prior to pseudo-labelling. Defaults to False.
        dist_align (Callable[[Tensor, Tensor], Tensor], optional):
            The distribution alignment function used. Expects a callable with arguments f(probs_ulbl, probs_lbl) which
            are normalised vectors with dimension matching num_classes. By default, the sslpack DistributionAlignment implementation is used.
        sup_loss_func (Callable[[Tensor, Tensor], Tensor], optional):
            a function with signature f(pred, true) to compute
            the loss on the supervised batch. Defaults to torch cross_entropy
        unsup_loss_func (Callable[[Tensor, Tensor, Tensor], Tensor], optional):
            a function with signature f(pred, true, mask) compute the loss on the unsupervised batch for only unmasked examples.
            Defaults to sslpack's masked cross entropy
        device (Union[device, str]): The torch device on which to store the FlexMatch state vectors
        """
    def __init__(self,
                 num_classes:int,
                 num_ulbl:int,
                 lambda_u:float=1,
                 conf_threshold:float=0.95,
                 use_warmup:bool=False,
                 concat:bool=True,
                 use_dist_align:bool=False,
                 dist_align:Optional[Callable[[Tensor, Tensor], Tensor]]=None,
                 sup_loss_func:Optional[Callable[[Tensor, Tensor], Tensor]]=None,
                 unsup_loss_func:Optional[Callable[[Tensor, Tensor, Tensor], Tensor]]=None,
                 device:Union[device, str]='cpu'):

        super().__init__()

        self.num_ulbl = num_ulbl
        self.num_classes = num_classes
        self.lambda_u = lambda_u
        self.conf_threshold = conf_threshold
        self.use_warmup = use_warmup
        self.concat = concat
        self.use_dist_align = use_dist_align
        if dist_align is None:
            self.dist_align = DistributionAlignment()
        else:
            self.dist_align = dist_align
        self.device = device

        self.sup_loss_func = ce if sup_loss_func is None else sup_loss_func
        self.unsup_loss_func = cel if unsup_loss_func is None else unsup_loss_func

        # unusued is set to the n+1 class index, useful for bincount
        self.UNUSED = num_classes
        # a vector tracking which unlabelled data have been used so far
        self.ulbl_preds = torch.ones((self.num_ulbl,), dtype=torch.long) * self.UNUSED
        # a vector tracking the number of predictions made for each class
        self.class_counts = torch.bincount(self.ulbl_preds, minlength=self.num_classes+1)
        self.class_thresholds = self.flex_threshold()

        self.to(device)

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

    def forward(self,
                model:nn.Module,
                lbl_batch:dict,
                ulbl_batch:dict,
                log_func:Optional[Callable[[dict], None]]=None):
        """
        Performs a forward pass of FlexMatch

        Args:
            model (nn.Module):
                The classification model
            lbl_batch (dict):
                A dictionary containing the labelled batch. Expects keys "weak", "strong" for
                augmented features, "y" for labels
            ubl_batch (dict):
                A dictionary with unlabelled data with keys. Expects keys "weak", "strong" for
                augmented features, and "idx" for data indices
            log_func (Callable[[dict], None]):
                A function which accepts a dictionary containing training information for the batch.
                The keys in the dictionary are "sup_loss", "unsup_loss", "total_loss",
                "confidences", "pseudo_labels", "mask", "class_count"
        """
        # model outputs on labelled/unlabelled examples with augmentations
        o_lbl_w, o_ulbl_w, o_ulbl_s = self._model_outputs(model, lbl_batch, ulbl_batch)

        probs_lbl_w, probs_ulbl_w = o_lbl_w.softmax(dim=1), o_ulbl_w.softmax(dim=1)

        if self.use_dist_align is True:
            probs_ulbl_w = self.dist_align(probs_ulbl_w, probs_lbl_w)

        confs, pseudo_labels, mask = self.flex_mask(probs_ulbl_w)
        self.update(confs, pseudo_labels, ulbl_batch["idx"])

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
                "mask": mask.detach().cpu(),
                "class_count": self.class_counts,
                "class_thresholds": self.class_thresholds.detach.cpu()
            })

        return total_loss

    @torch.no_grad()
    def flex_threshold(self):
        """
        Compute the class-wise confidence thresholds
        """
        counts = self.class_counts # num_class element vector
        if torch.argmax(counts[:self.num_classes]) < counts[self.UNUSED] and self.use_warmup:
            # unused data dominate
            beta = counts[:self.num_classes] / torch.max(counts)
        else:
            # used data dominate
            beta = counts[:self.num_classes] / torch.max(counts[:self.num_classes])
        return beta * self.conf_threshold

    @torch.no_grad()
    def flex_mask(self, probs):
        """
        Generate a flexmatch mask using classwise thresholds
        """
        confs, pseudo_labels = torch.max(probs, dim=1)
        # get the threshold for each observation in the batch
        threshold = self.class_thresholds[pseudo_labels]
        mask = confs.ge(threshold)
        return confs, pseudo_labels, mask

    @torch.no_grad()
    def update(self, confs, pseudo_labels, idxs):
        """
        Update counts and predictions on unlabelled data
        """
        select = confs.ge(self.conf_threshold)
        self.ulbl_preds[idxs[select == 1]] = pseudo_labels[select == 1]
        self.class_counts = torch.bincount(self.ulbl_preds,minlength=self.num_classes+1)
        self.class_thresholds = self.flex_threshold()

    def to(self, device):
        self.class_counts = self.class_counts.to(device)
        self.ulbl_preds = self.ulbl_preds.to(device)
        self.class_thresholds = self.class_thresholds.to(device)

    def reset(self):
        self.ulbl_preds = torch.ones((self.num_ulbl,), dtype=torch.long) * self.UNUSED
        self.class_counts = torch.bincount(self.ulbl_preds, minlength=self.num_classes+1)
        self.class_thresholds = self.flex_threshold()
