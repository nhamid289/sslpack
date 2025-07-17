import torch


def _threshold_mask(model, x, threshold, max_labels):
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confs, pseudo_labels = torch.max(probs, dim=1)
        mask = confs.ge(threshold)
        if max_labels is not None:
            _, indices = torch.topk(confs, min(max_labels, len(confs)))
            keep = torch.zeros_like(mask, dtype=torch.bool)
            keep[indices] = True
            mask &= keep

    return confs, pseudo_labels, mask
