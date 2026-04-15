"""Learning rate scheduler."""
import math

import mindspore as ms
from mindspore import ops
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class OneCycleLR(LearningRateSchedule):
    """One cycle learning rate scheduler."""
    def __init__(self, lr_max, total_steps, pct_start=0.3, init_factor=25, final_factor=10000.):
        super(OneCycleLR, self).__init__()
        self.lr_max = lr_max
        self.lr_init = lr_max/init_factor
        self.lr_final = lr_max/final_factor
        self.total_steps = total_steps
        self.step_mid = int(pct_start*total_steps)

    def construct(self, global_step):
        """Create learning rate."""
        lr_max = self.lr_max
        lr_init = self.lr_init
        lr_final = self.lr_final
        total_steps = self.total_steps
        step_mid = self.step_mid

        global_step = global_step.astype(ms.float32)
        if global_step <= step_mid:
            cos_factor = 1 + ops.cos(math.pi*(global_step/step_mid - 1))
            lr = .5*(lr_max - lr_init)*cos_factor + lr_init
        else:
            cos_factor = 1 + ops.cos(math.pi*((global_step - step_mid)/(total_steps - step_mid)))
            lr = .5*(lr_max - lr_final)*cos_factor + lr_final
        return lr
