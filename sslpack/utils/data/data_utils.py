import numpy as np


def stratify_lbl_ulbl(
        X,
        y,
        lbls_per_class,
        ulbls_per_class=None,
        seed=None,
    ):
    """
    A function to split features and labels into separate labelled and
    unlabelled sets.

    Args:
        X (arraylike):
            An array of features
        y (arraylike):
            An array of labels corresponding to the features
        lbls_per_class (int or list):
            The number of instances per class to be labelled.
            If an integer, that number of labels are sampled for every class.
            Alternatively, provide a list of size matching the number of classes.
            Each element corresponds to the number of labels for each class.
        ulbls_per_class (int or list, optional):
            The number of instances per class to be taken as unlabelled.
            If an integer, that number of labels are sampled for every class.
            Alternatively, provide a list of size matching the number of classes.
            Each element corresponds to the number of unlabelled instances for each class.
        seed (int, optional):
            The seed for the random selection of instances

    Returns:
        Returns a 4-tuple (X_lbl, y_lbl, X_ulbl, y_ulbl)
    """
    lbls = []
    ulbls = []

    if seed is not None:
        np.random.seed(seed)

    for (i, label) in enumerate(np.unique(y)):
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)
        # take the first lbls_per_class from shuffled indices
        if isinstance(lbls_per_class, list):
            lbls.extend(idx[:lbls_per_class[i]])
        else:
            lbls.extend(idx[:lbls_per_class])

        if ulbls_per_class is None:
            ulbls.extend(idx[lbls_per_class:])
        elif isinstance(ulbls_per_class, list):
            ulbls.extend(idx[lbls_per_class : lbls_per_class + ulbls_per_class[i]])
        else:
            ulbls.extend(idx[lbls_per_class : lbls_per_class + ulbls_per_class])

    return (
        [X[i] for i in lbls],
        [y[i] for i in lbls],
        [X[i] for i in ulbls],
        [y[i] for i in ulbls],
    )

def stratify_lbl_ulbl_idx(
        y,
        lbls_per_class,
        ulbls_per_class=None,
        seed=None,
    ):
    """
    A function to split features and labels into separate labelled and
    unlabelled sets.

    Args:
        y (arraylike):
            An array of labels
        lbls_per_class (int or list):
            The number of instances per class to be labelled.
            If an integer, that number of labels are sampled for every class.
            Alternatively, provide a list of size matching the number of classes.
            Each element corresponds to the number of labels for each class.
        ulbls_per_class (int or list, optional):
            The number of instances per class to be taken as unlabelled.
            If an integer, that number of labels are sampled for every class.
            Alternatively, provide a list of size matching the number of classes.
            Each element corresponds to the number of unlabelled instances for each class.
        seed (int, optional):
            The seed for the random selection of instances

    Returns:
        Returns a pair of lists corresponding to the labelled and unlabelled indices
    """
    lbls = []
    ulbls = []

    if seed is not None:
        np.random.seed(seed)

    for (i, label) in enumerate(np.unique(y)):
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)
        # take the first lbls_per_class from shuffled indices
        if isinstance(lbls_per_class, list):
            lbls.extend(idx[:lbls_per_class[i]])
        else:
            lbls.extend(idx[:lbls_per_class])

        if ulbls_per_class is None:
            ulbls.extend(idx[lbls_per_class:])
        elif isinstance(ulbls_per_class, list):
            ulbls.extend(idx[lbls_per_class : lbls_per_class + ulbls_per_class[i]])
        else:
            ulbls.extend(idx[lbls_per_class : lbls_per_class + ulbls_per_class])

    return lbls, ulbls
