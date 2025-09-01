import torch
import torch.nn as nn

class DistributionAlignment(nn.Module):
    def __init__(self, prior=None, momentum=0.999):
        super(DistributionAlignment, self).__init__()
        self.prior = prior
        self.momentum = momentum
        self.num_classes = None
        self.model_dist = None

        if self.prior is not None:
            self.target_dist = self.prior
        else:
            self.target_dist = None

    def forward(self, probs_ulb, probs_lb):
        if self.num_classes is None:
            self.num_classes = probs_ulb.shape[1]

        if self.model_dist is None:
            self.model_dist = torch.ones(self.num_classes, device=probs_ulb.device)/self.num_classes

        if self.prior is None:
            if self.target_dist is None:
                self.target_dist = torch.ones(self.num_classes, device=probs_ulb.device)/self.num_classes
            else:
                self.target_dist = self.momentum*self.target_dist + (1-self.momentum)*probs_lb.mean(dim=0)

        #---------------------------------
        self.model_dist = self.momentum*self.model_dist + (1-self.momentum)*probs_ulb.mean(dim=0)
        out = (probs_ulb + 1e-6) * (self.target_dist + 1e-6) / (self.model_dist + 1e-6)
        return out/out.sum(dim=1, keepdim=True)
