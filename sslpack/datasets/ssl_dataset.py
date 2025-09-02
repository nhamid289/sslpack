
class SSLDataset:
    """
    An interface to define SSL datasets.

    An SSLDataset should contain
        - a labelled dataset (get_lbl_dataset())
        - an unlabelled dataset (get_ulbl_dataset())
        - a validation set (get_val_dataset())
        - an evaluation set (get_eval_dataset())

    It is expected these datasets are using sslpack implementations. See `sslpack.utils.data.dataset`
    """

    def __init__(self):
        self.lbl_dataset = None
        self.ulbl_dataset = None
        self.eval_dataset = None
        self.val_dataset = None

    def get_lbl_dataset(self):
        """
        Get the labelled part of this SSLDataset
        """
        return self.lbl_dataset

    def get_ulbl_dataset(self):
        """
        Get the unlabelled part of this SSLDataset.
        """
        return self.ulbl_dataset

    def get_eval_dataset(self):
        """
        Get the evaluation part of this SSLDataset
        """
        return self.eval_dataset

    def get_val_dataset(self):
        """
        Get the validation part of this SSLDataset
        """
        return self.val_dataset
