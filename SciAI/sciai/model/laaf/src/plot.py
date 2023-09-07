"""laaf plot"""
import matplotlib.pyplot as plt
import numpy as np

from sciai.utils.plot_utils import newfig, savefig


def plot_train(figures_path, sol, x, y):
    """plot training results"""
    solution = np.concatenate(sol, axis=1)
    newfig(1.0, 1.1)
    x, y = x[0:-1].asnumpy(), y[0:-1].asnumpy()
    plt.plot(x, y, 'k-', label='Exact')
    plt.plot(x, solution[0:-1, -1], 'yx-', label='Predicted at Iter = 15000')
    plt.plot(x, solution[0:-1, 1], 'b-.', label='Predicted at Iter = 8000')
    plt.plot(x, solution[0:-1, 0], 'r--', label='Predicted at Iter = 2000')
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    plt.legend(loc='upper left')
    savefig(f'{figures_path}/laaf')


def plot_eval(figures_path, sol, x, y):
    """plot evaluation results"""
    newfig(1.0, 1.1)
    x, y = x[0:-1].asnumpy(), y[0:-1].asnumpy()
    plt.plot(x, y, 'k-', label='Exact')
    plt.plot(x, sol.asnumpy()[0:-1], 'yx-', label='Predicted at Iter = 15001')
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    plt.legend(loc='upper left')
    savefig(f'{figures_path}/laaf_val')
