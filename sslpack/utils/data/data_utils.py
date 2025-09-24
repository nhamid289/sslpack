import numpy as np


def split_lb_ulb_balanced(
    X,
    y,
    lbls_per_class,
    ulbls_per_class=None,
    return_idx=False,
    seed=None,
):
    """
    A function to split features and labels into separate labelled and
    unlabelled sets.

    Args:
        X: the features
        y: the labels
        lbls_per_class: The number of samples per class to be labelled
        ulbls_per_class: The number of samples per class to be unlabelled.
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

    for (i, label) in enumerate(np.unique(y)):
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)
        # take the first lbls_per_class from shuffled indices
        lbls.extend(idx[:lbls_per_class])
        if ulbls_per_class is None:
            ulbls.extend(idx[lbls_per_class:])
        elif isinstance(ulbls_per_class, list):
            ulbls.extend(idx[lbls_per_class : lbls_per_class + ulbls_per_class[i]])
        else:
            ulbls.extend(idx[lbls_per_class : lbls_per_class + ulbls_per_class])

    if return_idx:
        return lbls, ulbls

    return (
        [X[i] for i in lbls],
        [y[i] for i in lbls],
        [X[i] for i in ulbls],
        [y[i] for i in ulbls],
    )
