import torch
from torch.nn.functional import cross_entropy as ce

from sslpack.algorithms import Algorithm
from sslpack.utils.criterions import ce_consistency_loss as cel

class FlexMatch(Algorithm):
    """ An implementation of FlexMatch (http://arxiv.org/abs/2110.08263)

    By default the algorithm uses cross entropy loss for the supervised part,
    and cross entropy consistency loss for the unsupervised part.
    """

    def __init__(self, num_classes, num_ulbl, lambda_u=1, conf_threshold=0.95, use_warmup=False, concat=True,
                 dist_align = None,  sup_loss_func=None, unsup_loss_func=None,
                 device = 'cpu'):
        """
        Initialise a FlexMatch algorithm. FlexMatch requires access to data indices for unlabelled examples
        to track the curriculum. This algorithm has a state, so must be reinitialised or reset() to clear
        the state for training a new model.

        Args:
            num_classes: The number of classes in the data
            num_ulbl: The total number of unlabelled examples in the training data
            lambda_u: The weight of the unlabelled loss in the total loss.
            conf_threshold: the confidence threshold for pseudo-labels
            use_warmup: If true, use threshold warmup when unused data dominate
            concat: If false, a separate forward pass is performed for unlabelled and labelled batches. Otherwise, the data is concanted and a single forward pass is performed.
            dist_align: A function that accepts a probability distribution vector for alignment. If none, no alignment is applied.
            sup_loss_func: a function with signature f(pred, true) to compute
                the loss on the supervised batch. Default: cross entropy
            unsup_loss_func: a function with signature f(pred, true, mask) to
                compute the loss on the unsupervised batch: Default: masked cross entropy
            device: the device to store the state vectors
        """
        super().__init__()

        self.num_ulbl = num_ulbl
        self.num_classes = num_classes
        self.lambda_u = lambda_u
        self.conf_threshold = conf_threshold
        self.use_warmup = use_warmup
        self.concat = concat
        self.dist_align = dist_align
        self.device = device

        self.sup_loss_func = ce if sup_loss_func is None else sup_loss_func
        self.unsup_loss_func = cel if unsup_loss_func is None else unsup_loss_func

        # unusued is set to the n+1 class index, useful for bincount
        self.UNUSED = num_classes + 1
        # a vector tracking which unlabelled data have been used so far
        self.ulbl_preds = torch.ones((self.num_ulbl,), dtype=torch.long) * self.UNUSED
        # a vector tracking the number of predictions made for each class
        self.class_counts = torch.bincount(self.ulbl_preds, minlength=self.UNUSED+1)

        self.to(device)

    def forward(self, model, lbl_batch, ulbl_batch, log_func=None):
        """
        Performs a forward pass of FixMatch

        Args:
            model: The predictor model
            lbl_batch: A dictionary containing the labelled batch. Expects keys
                "weak", "strong" for augmented features, "y" for labels
            ubl_batch: A dictionary with unlabelled data with keys. Expects keys
                "weak", "strong" for augmented features, and "idx" for data indices
            log_func: A function which accepts a dictionary containing some
                training information
        """
        if self.concat is True:
            lbl_size = lbl_batch["weak"].size(0)
            ulbl_size = ulbl_batch["weak"].size(0)
            x = torch.concat([lbl_batch["weak"], ulbl_batch["weak"], ulbl_batch["strong"]])
            out = model(x)
            out_lbl_weak = out[:lbl_size] # unconcat the outputs
            out_ulbl_weak = out[lbl_size: lbl_size + ulbl_size]
            out_ulbl_strong = out[lbl_size + ulbl_size:]
        else:
            out_lbl_weak = model(lbl_batch["weak"])
            out_ulbl_strong = model(ulbl_batch["strong"])
            with torch.no_grad():
                out_ulbl_weak = model(ulbl_batch["weak"])

        probs_ubl_weak = torch.softmax(out_ulbl_weak, dim=1)

        if self.dist_align is not None:
            probs_ubl_weak = self.dist_align(probs_ubl_weak)

        confs, pseudo_labels, mask = self.flex_mask(probs_ubl_weak)
        self.update(confs, pseudo_labels, ulbl_batch["idx"])

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
                "mask": mask.detach().cpu(),
                "class_count": self.class_counts,
            })

        return total_loss

    @torch.no_grad()
    def flex_threshold(self):
        """
        Compute the class-wise confidence thresholds
        """
        counts = self.class_counts # num_class element vector
        if torch.argmax(counts) < counts[self.UNUSED] and self.use_warmup:
            # unused data dominate
            beta = counts[:self.UNUSED] / torch.max(counts)
        else:
            # used data dominate
            beta = counts[:self.UNUSED] / torch.max(counts[:self.UNUSED])
        return beta * self.conf_threshold

    @torch.no_grad()
    def flex_mask(self, probs):
        """
        Generate a flexmatch mask using classwise thresholds
        """
        confs, pseudo_labels = torch.max(probs, dim=1)
        class_thresholds = self.flex_threshold()
        # get the threshold for each observation in the batch
        threshold = class_thresholds[pseudo_labels]
        mask = confs.ge(threshold)
        return confs, pseudo_labels, mask

    @torch.no_grad()
    def update(self, confs, pseudo_labels, idxs):
        """
        Update counts and predictions on unlabelled data
        """
        select = confs.ge(self.conf_threshold)
        self.ulbl_preds[idxs[select == 1]] = pseudo_labels[select == 1]
        self.class_counts = torch.bincount(self.ulbl_preds,minlength=self.UNUSED+1)

    def to(self, device):
        self.class_counts = self.class_counts.to(device)
        self.ulbl_preds = self.ulbl_preds.to(device)

    def reset(self):
        self.ulbl_preds = torch.ones((self.num_ulbl,), dtype=torch.long) * self.UNUSED
        self.class_counts = torch.bincount(self.ulbl_preds, minlength=self.UNUSED+1)
