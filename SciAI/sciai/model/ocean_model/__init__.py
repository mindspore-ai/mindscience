"""init for ocean model"""
from train import main as main_train
from src.utils import prepare
from eval import main as main_eval

__all__ = ["main_train", "main_eval", "prepare"]
