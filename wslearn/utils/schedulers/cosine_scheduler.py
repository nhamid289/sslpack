import math
from torch.optim.lr_scheduler import LambdaLR

class CosineSchedulerWithWarmup(LambdaLR):

    def __init__(self, optimizer, num_train_iter, num_warmup_iter=0, num_cycles=7/16, last_epoch = -1):

        self.num_train_iter = num_train_iter
        self.num_warmup_iter = num_warmup_iter
        self.num_cycles = num_cycles

        super().__init__(optimizer, self._lr_lambda, last_epoch)

    def _lr_lambda(self, current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < self.num_warmup_iter:
            _lr = float(current_step) / float(max(1, self.num_warmup_iter))
        else:
            num_cos_steps = float(current_step - self.num_warmup_iter)
            num_cos_steps = num_cos_steps / float(max(1, self.num_train_iter - self.num_warmup_iter))
            _lr = max(0.0, math.cos(math.pi * self.num_cycles * num_cos_steps))
        return _lr