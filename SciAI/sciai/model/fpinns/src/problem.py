"""problem structure"""
from abc import abstractmethod, ABC

import mindspore as ms
from matplotlib import pyplot as plt
from sciai.utils import data_type_dict_amp, data_type_dict_np


class Problem(ABC):
    """Problem definition"""
    def __init__(self, args):
        self.dtype = data_type_dict_amp.get(args.amp_level, ms.float32)
        self.dtype_np = data_type_dict_np.get(self.dtype)
        self.epochs = args.epochs

        self.num_domain = args.num_domain
        self.num_initial = args.num_initial
        self.x_range = args.x_range
        self.t_range = args.t_range
        self.num_boundary = args.num_boundary

        self.save_fig = args.save_fig
        self.figures_path = args.figures_path

        self.loss = []
        self.steps = []

    @abstractmethod
    def setup_train_cell(self, args, net):
        pass

    @abstractmethod
    def setup_networks(self, args):
        pass

    @abstractmethod
    def train(self, train_cell):
        pass

    @abstractmethod
    def predict(self, network, *inputs):
        pass

    @abstractmethod
    def func(self, x, t):
        pass

    @abstractmethod
    def plot_result(self, x_test, t_test, y_test, y_res):
        pass

    @abstractmethod
    def generate_data(self, num):
        pass

    def plot_train_process(self):
        plt.figure(1)
        plt.semilogy(self.steps, self.loss, label="Train loss")
        plt.xlabel("# Steps")
        plt.legend()
        plt.savefig(f"{self.figures_path}/loss_history_ms.png")
