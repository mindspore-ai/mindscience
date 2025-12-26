# pylint: disable=invalid-name
"""
Embedding functions.
"""
from mindspore import ops
import mindspore.numpy as mnp

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps (Tensor): Timestep values to embed. Shape (batch_size,).
        dim (int): Dimension of the output embeddings.
        max_period (int, optional): Maximum period for sine/cosine functions. Default: 10000.
    
    Returns:
        Tensor: Embedded timesteps with shape (batch_size, dim).
    """
    half = dim // 2
    freqs = ops.exp(
        -mnp.log(max_period) * mnp.arange(start=0, stop=half, dtype=mnp.float32) / half
    )
    args = timesteps[:, None].astype(mnp.float32) * freqs[None]
    embedding = ops.concat([ops.cos(args), ops.sin(args)], axis=-1)
    if dim % 2:
        embedding = ops.concat([embedding, ops.zeros_like(embedding[:, :1])], axis=-1)
    return embedding
    