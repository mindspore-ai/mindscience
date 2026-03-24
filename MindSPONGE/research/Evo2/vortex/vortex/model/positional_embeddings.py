"""
Armin Thomas, Jan 2023.  Modified by Eric Nguyen.

Wrappers for linearly interpolated rope embeddings to use inside of MHA layers of Flash Attn.

"""

import mindspore as ms
from mindspore import mint, nn, Tensor
from einops import rearrange
from vortex.model.rotary import RotaryEmbedding


# simple wrapper for flash-attn RoPE with linear scaling:
class LinearlyScaledRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        scaling_factor: float = 1.0,
        base: float = 10_000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
    ):
        super().__init__(
            dim=dim,
            base=base,
            interleaved=interleaved,
            scale_base=scale_base,
            pos_idx_in_fp32=pos_idx_in_fp32,
        )
        self._linear_scaling_factor = scaling_factor

    # adpated from: https://github.com/Dao-AILab/flash-attention/blob/43ceab630bc6c27712428da5a33fc9cb5c369d91/flash_attn/layers/rotary.py#L368
    def _update_cos_sin_cache(self, seqlen：int, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > getattr(self, "_seq_len_cached", 0)
            or self._cos_cached is None
            or self._cos_cached.dtype != dtype
            or (self.training and getattr(self._cos_cached, "is_inference", lambda: False)())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = mint.arange(seqlen, dtype=ms.float32)
                # linear scaling:
                t = t / self._linear_scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                inv_freq = (
                    self._compute_inv_freq()
                    if self.inv_freq.dtype != ms.float32
                    else self.inv_freq
                )
            else:
                t = mint.arange(seqlen, dtype=self.inv_freq.dtype)
                # linear scaling:
                t = t / self._linear_scaling_factor
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = mint.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = mint.cos(freqs).to(dtype)
                self._sin_cached = mint.sin(freqs).to(dtype)
            else:
                power = (
                    mint.arange(seqlen, dtype=self.scale.dtype) - seqlen // 2
                ) / self.scale_base
                scale = self.scale ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (mint.cos(freqs) * scale).to(dtype)
                self._sin_cached = (mint.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (mint.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (mint.sin(freqs) / scale).to(dtype)


# swap out RoPE of existing mha:
def swap_mha_rope(
    mha,
    new_rope: type = LinearlyScaledRotaryEmbedding,
    kwargs_new_rope: dict | None = None,
):
    # determine mha dtype and device:
    if mha.cross_attn:
        dtype = mha.Wq.weight.dtype
    else:
        dtype = mha.Wqkv.weight.dtype
    # determine RoPE settings:
    kwargs_old_rope = dict(
        dim=mha.rotary_emb.dim,
        base=mha.rotary_emb.base,
        interleaved=getattr(mha.rotary_emb, "interleaved", False),
        scale_base=getattr(mha.rotary_emb, "scale_base", None),
        pos_idx_in_fp32=getattr(mha.rotary_emb, "pos_idx_in_fp32", True),
    )
    # delete old RoPE:
    del mha.rotary_emb
    # create new RoPE:
    kwargs_new = kwargs_new_rope or {"scaling_factor": 1.0}

    # attach new RoPE to mha:
    mha.rotary_emb = new_rope(**kwargs_new, **kwargs_old)
    # make new sure RoPE is correctly registered:
    assert isinstance(mha.rotary_emb, new_rope)
    return mha
