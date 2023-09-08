"""problem"""
from abc import abstractmethod, ABC


class Problem(ABC):
    @abstractmethod
    def plot(self, data, net, figure_path):
        pass

    @abstractmethod
    def init_data(self, args):
        pass
