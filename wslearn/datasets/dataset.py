import torch
import numpy as np

class Dataset():

    def __init__(self):
        self.lbl_dataset = None
        self.ulbl_dataset = None
        self.eval_dataset = None

    def get_lbl_dataset(self):
        return self.lbl_dataset

    def get_ulbl_dataset(self):
        return self.ulbl_dataset

    def get_eval_dataset(self):
        return self.eval_dataset





