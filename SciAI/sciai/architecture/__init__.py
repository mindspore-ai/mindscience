"""sciai architecture"""
from .activation import Swish, SReLU, get_activation, AdaptActivation
from .basic_block import MLP, MLPAAF, MLPShortcut, MSE, SSE, FirstOutputCell, NoArgNet, Normalize
from .neural_operators import FNO1D, FNO2D, FNO3D, KNO1D, KNO2D, PDENet
from .transformer import ViT

__all__ = []
__all__.extend(["Swish", "SReLU", "get_activation", "AdaptActivation"])
__all__.extend(["MLP", "MLPAAF", "MLPShortcut", "MSE", "SSE", "FirstOutputCell", "NoArgNet", "Normalize"])
__all__.extend(["FNO1D", "FNO2D", "FNO3D", "KNO1D", "KNO2D", "PDENet"])
__all__.extend(["ViT"])
