class Algorithm:

    def __init__(self):
        pass



    def forward(self, model, lbl_batch:dict, ulbl_batch:dict, log_func=None):
        """
        forward specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model
        # log
        # return loss
        raise NotImplementedError

