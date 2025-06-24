import numpy as np

def split_lb_ulb_balanced(X, y, num_lbl, num_ulbl = None,
                 lbl_idx=None, ulbl_idx=None, lbl_in_ulbl=True,
                 return_idx = False, seed=None):
    """
    A function to split features and labels into separate labelled and
    unlabelled sets.

    Args:
        X: the features
        y: the labels
        num_classes: The number of target classes
        num_lbl: The number of samples per class to be labelled
        num_ulbl: The number of samples per class to be unlabelled.
            If left unspecified, all remaining unlabelled data is taken
        lbl_idx: The specific indices to include in labelled data.
        ulbl_indx: The specific indices to include in unlabelled data.

    Returns
        If return_idx is True:
            Returns a tuple of lists containing the labelled and unlabelled
            indices
        Else:
            Returns a 4-tuple containing the labelled features and labels,
            and unlabelled features and labels
    """
    lbls = []
    ulbls = []
    if seed is not None:
        np.random.seed(seed)

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)
        # take the first num_lbl from shuffled indices
        lbls.extend(idx[:num_lbl])
        if num_ulbl is None:
            ulbls.extend(idx[num_lbl:])
        else:
            ulbls.extend(idx[num_lbl: num_lbl + num_ulbl])

    if return_idx:
        return lbls, ulbls

    return ([X[i] for i in lbls], [y[i] for i in lbls],
            [X[i] for i in ulbls], [X[i] for i in ulbls])
