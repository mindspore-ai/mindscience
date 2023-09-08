"""init for pinns"""
from train import main as main_train
from eval import main as main_eval
from src.utils import prepare

__all__ = ["main_train", "main_eval", "prepare"]
