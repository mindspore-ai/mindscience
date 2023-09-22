"""init for inversion net"""
from train import main as main_train
from eval import main as main_eval
from src.process import prepare

__all__ = ["main_train", "main_eval", "prepare"]
