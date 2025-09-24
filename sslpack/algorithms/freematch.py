import torch
from torch.nn.functional import cross_entropy as ce

from sslpack.algorithms import Algorithm
from sslpack.algorithms.utils import DistributionAlignment
from sslpack.utils.criterions import ce_consistency_loss as cel

from torch import Tensor, device, nn
from typing import Optional, Callable, Union

class FreeMatch(Algorithm):
    """ An implementation of FlexMatch (https://arxiv.org/pdf/2205.07246)

    FreeMatch uses pseudo-labelling and augmentation anchoring like FixMatch. Like FlexMatch, it uses an adaptative class-wise confidence threshold, but adds a class fairness penalty term to encourage diverse predictions during the early training stage.

    Args:
        num_classes (int):
            The number of classes. Expects positive integer > 0.
        lambda_u (float, optional):
            The weight of the unlabelled loss in the total loss. Expects non-negative real >= 0. Defaults to 1.
        lambda_f (float, optional):
            The weight of the fairness loss in the total loss. Expects non-negative real >= 0. Defaults to 1.
        threshold_decay (float, optional)
            The exponential decay coefficient to use for threshold decay. Expects non-negative real in [0, 1]. Defaults to 0.999
        clip_threshold (tuple[float, float], optional):
            The (lower, upper) clip for the global threshold on each iteration. If None, no clipping is applied.
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
                 lambda_u:float=1,
                 lambda_f:float=0.01,
                 threshold_decay:float=0.999,
                 clip_threshold:Optional[tuple[float, float]]=(0.0, 0.95),
                 concat:bool=True,
                 use_dist_align:bool=False,
                 dist_align:Optional[Callable[[Tensor, Tensor], Tensor]]=None,
                 sup_loss_func:Optional[Callable[[Tensor, Tensor], Tensor]]=None,
                 unsup_loss_func:Optional[Callable[[Tensor, Tensor, Tensor], Tensor]]=None,
                 device:Union[device, str]='cpu'):

        super().__init__()


        self.num_classes = num_classes
        self.lambda_u = lambda_u
        self.lambda_f = lambda_f
        self.threshold_decay = threshold_decay
        self.clip_threshold = clip_threshold
        self.concat = concat
        self.use_dist_align = use_dist_align
        if dist_align is None:
            self.dist_align = DistributionAlignment()
        else:
            self.dist_align = dist_align
        self.device = device

        self.sup_loss_func = ce if sup_loss_func is None else sup_loss_func
        self.unsup_loss_func = cel if unsup_loss_func is None else unsup_loss_func

        self.global_threshold = 1 / self.num_classes
        self.class_probs = torch.ones((self.num_classes)) / self.num_classes
        self.pred_hist = torch.ones((self.num_classes)) / self.num_classes
        self.class_thresholds = self.class_probs / torch.max(self.class_probs, dim=-1)[0]

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

        with torch.no_grad():
            confs_w, pseudo_labels_w = torch.max(probs_ulbl_w, dim=1)
            self._update_thresholds(probs_ulbl_w, confs_w, pseudo_labels_w)
            mask = self._classwise_mask(confs_w, pseudo_labels_w)


        sup_loss = self.sup_loss_func(o_lbl_w, lbl_batch["y"])
        unsup_loss = self.unsup_loss_func(o_ulbl_s, pseudo_labels_w, mask)
        if mask.sum() == 0:
            fairness_loss = 0.0
        else:
            fairness_loss = self._fairness_loss(o_ulbl_s, mask)

        total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_f * fairness_loss

        if log_func is not None:
            log_func({
                "sup_loss": sup_loss.item(),
                "unsup_loss": unsup_loss.item(),
                "fairness_loss": fairness_loss.item(),
                "total_loss": total_loss.item(),
                "confidences": confs_w.detach().cpu(),
                "pseudo_labels": pseudo_labels_w.detach().cpu(),
                "mask": mask.detach().cpu(),
                "global_threshold": self.global_threshold.item(),
                "class_thresholds": self.class_thresholds.detach().cpu()
            })

        return total_loss

    @torch.no_grad()
    def _classwise_mask(self, confidences, pseudo_labels):
        mask = confidences.ge(self.global_threshold * self.class_thresholds[pseudo_labels])
        return mask

    def _update_ema(self, x, y):
        return x * self.threshold_decay + (1 - self.threshold_decay) * y

    @torch.no_grad()
    def _update_thresholds(self, probs, confs, pseudos):

        self.global_threshold = self._update_ema(self.global_threshold, confs.mean())

        if self.clip_threshold is not None:
            self.global_threshold = torch.clip(self.global_threshold, self.clip_threshold[0], self.clip_threshold[1])

        self.class_probs = self._update_ema(self.class_probs, probs.mean(dim=0))
        # the count of predictions for each class
        preds = torch.bincount(pseudos.reshape(-1), minlength=self.class_probs.shape[0])

        self.pred_hist = self._update_ema(self.pred_hist, preds)
        self.class_thresholds = (self.class_probs / torch.max(self.class_probs, dim=-1)[0]) * self.global_threshold

    def _replace_inf_to_zero(self, val):
        val[val == float('inf')] = 0.0
        return val

    def _fairness_loss(self, o_ulbl_s, mask):

        probs_ulbl_s = o_ulbl_s.softmax(dim=1)
        class_probs_s = (probs_ulbl_s * mask.unsqueeze(1)).mean(dim=0, keepdim=True)

        _, pseudos_s = torch.max(probs_ulbl_s, dim=1)
        preds_s = torch.bincount(pseudos_s, minlength=o_ulbl_s.shape[1])
        class_hist_s = preds_s / preds_s.sum()

        # modulate prob model
        class_probs_w, class_hist_w = self.class_probs.unsqueeze(0), self.pred_hist.unsqueeze(0)
        prob_scaler_w = self._replace_inf_to_zero(1 / class_hist_w)
        mod_prob_w = class_probs_w * prob_scaler_w
        mod_prob_w = mod_prob_w / mod_prob_w.sum(dim=-1, keepdim=True)

        # modulate mean prob
        prob_scaler_s = self._replace_inf_to_zero(1 / class_hist_s)
        mod_prob_s = class_probs_s * prob_scaler_s
        mod_prob_s = mod_prob_s / mod_prob_s.sum(dim=-1, keepdim=True)

        loss = ce(mod_prob_w, mod_prob_s)
        return loss

    def to(self, device):
        self.class_probs = self.class_probs.to(device)
        self.pred_hist = self.pred_hist.to(device)

    def reset(self):
        self.class_probs = torch.ones((self.num_classes)) / self.num_classes
        self.pred_history = torch.ones((self.num_classes)) / self.num_classes
        self.to(self.device)



