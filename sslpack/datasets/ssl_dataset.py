
class SSLDataset:
    """
    An interface to define SSL datasets
    """

    def __init__(self):
        self.lbl_dataset = None
        self.ulbl_dataset = None
        self.eval_dataset = None
        self.val_dataset = None

    def get_lbl_dataset(self):
        return self.lbl_dataset

    def get_ulbl_dataset(self):
        return self.ulbl_dataset

    def get_eval_dataset(self):
        return self.eval_dataset

    def get_val_dataset(self):
        return self.val_dataset
