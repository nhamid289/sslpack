import torch

def threshold_mask(probs, threshold, max_labels):
    with torch.no_grad():
        confs, pseudo_labels = torch.max(probs, dim=1)
        mask = confs.ge(threshold)
        if max_labels is not None:
            _, indices = torch.topk(confs, min(max_labels, len(confs)))
            keep = torch.zeros_like(mask, dtype=torch.bool)
            keep[indices] = True
            mask &= keep

    return confs, pseudo_labels, mask