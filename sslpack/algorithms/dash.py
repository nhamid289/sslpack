from sslpack.algorithms import FixMatch
from sslpack.algorithms.utils import threshold_mask

class Dash(FixMatch):

    def __init__(self, lambda_u=1, C=1.0001, gamma=1.01, rho_init=0.1,
                 num_update_iters=8,
                 concat=True, dist_align=None, max_pseudo_labels=None,
                 sup_loss_func=None, unsup_loss_func=None):



        self.C = C
        self.gamma = gamma
        self.iter = 0
        self.rho_init = rho_init

        super().__init__(lambda_u, C*rho_init, concat,
                         dist_align, max_pseudo_labels,
                         sup_loss_func, unsup_loss_func)


    def reset(self):
        self.iter = 0
        self.conf_threshold = self.conf_init

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

        if self.dist_align is not None:
            probs_ulbl_weak = self.dist_align(probs_ulbl_w, probs_lbl_w)

        # pseudo-labels are generated under no_grad()
        confs, pseudo_labels, mask = threshold_mask(probs_ulbl_weak,
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
