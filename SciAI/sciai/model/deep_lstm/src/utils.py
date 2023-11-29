"""function definition"""
import numpy as np


def adjust_learning_rate(base_lr, decay, step_total):
    """adjust learning rate"""
    lr_list = []
    for step in range(step_total):
        gamma = 1 + decay * step
        assert gamma != 0
        lr = base_lr * (1 / gamma)
        lr_list.append(lr)
    return np.array(lr_list)
