import mindspore as ms
from mindspore import mint, nn, Tensor, Parameter
from mindspore import ops as P
from einops import rearrange
from vortex.model.rotary import RotaryEmbedding  

# ------------------------------------------------------------------
#  线性缩放的 RoPE（MindSpore 版）
# ------------------------------------------------------------------
class LinearlyScaledRotaryEmbedding(RotaryEmbedding):
    """
    在已有 RotaryEmbedding 基础上，对 position index 做线性缩放：
        t' = t / scaling_factor
    其余逻辑与原版保持一致。
    """
    def __init__(
        self,
        dim: int,
        scaling_factor: float = 1.0,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base=None,
        pos_idx_in_fp32: bool = True,
    ):
        super().__init__(
            dim=dim,
            base=base,
            interleaved=interleaved,
            scale_base=scale_base,
            pos_idx_in_fp32=pos_idx_in_fp32,
        )
        self._linear_scaling_factor = scaling_factor

        # 缓存张量统一用 Parameter(requires_grad=False) 管理
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    # ----------------------------------------------------------
    #  重写缓存更新逻辑（全部用 MindSpore API）
    # ----------------------------------------------------------
    def _update_cos_sin_cache(self, seqlen: int, dtype=None):
        # 各类需要刷新缓存的条件
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen

            # 1. 生成 position index
            if self.pos_idx_in_fp32:
                t = mint.arange(seqlen, dtype=ms.float32)
                t = t / self._linear_scaling_factor
                inv_freq = (
                    self._compute_inv_freq()
                    if self.inv_freq.dtype != ms.float32
                    else self.inv_freq
                )
            else:
                t = mint.arange(seqlen, dtype=self.inv_freq.dtype)
                t = t / self._linear_scaling_factor
                inv_freq = self.inv_freq

            # 2. 计算 freqs = t @ inv_freq^T
            freqs = P.outer(t, inv_freq)          # 等价于 torch.outer

            # 3. 应用可选的 scale 因子（若使用 scale_base）
            if self.scale is None:
                cos = mint.cos(freqs).astype(dtype)
                sin = mint.sin(freqs).astype(dtype)
            else:
                power = (mint.arange(seqlen, dtype=self.scale.dtype) - seqlen // 2) \
                        / self.scale_base
                scale = self.scale ** rearrange(power, "s -> s 1")

                cos = (mint.cos(freqs) * scale).astype(dtype)
                sin = (mint.sin(freqs) * scale).astype(dtype)
                cos_k = (mint.cos(freqs) / scale).astype(dtype)
                sin_k = (mint.sin(freqs) / scale).astype(dtype)

                self._cos_k_cached = Parameter(cos_k, requires_grad=False)
                self._sin_k_cached = Parameter(sin_k, requires_grad=False)

            # 4. 缓存结果
            self._cos_cached = Parameter(cos, requires_grad=False)
            self._sin_cached = Parameter(sin, requires_grad=False)


# ------------------------------------------------------------------
#  把已有 MHA 中的 RoPE 替换成新的 LinearlyScaledRotaryEmbedding
# ------------------------------------------------------------------
def swap_mha_rope(mha,
                  new_rope: nn.Cell = LinearlyScaledRotaryEmbedding,
                  kwargs_new_rope: dict = None):
    """
    将 mha 中的 rotary_emb 动态替换为支持线性缩放的新 RoPE。
    """
    # 1. 提取旧 RoPE 的关键参数
    kwargs_old_rope = dict(
        dim=mha.rotary_emb.dim,
        base=mha.rotary_emb.base,
        interleaved=getattr(mha.rotary_emb, "interleaved", False),
        scale_base=getattr(mha.rotary_emb, "scale_base", None),
        pos_idx_in_fp32=getattr(mha.rotary_emb, "pos_idx_in_fp32", True),
    )

    # 2. 删除旧 RoPE
    del mha.rotary_emb

    # 3. 创建并挂载新 RoPE
    kwargs_new_rope = kwargs_new_rope or {"scaling_factor": 1.0}
    mha.rotary_emb = new_rope(**kwargs_old_rope, **kwargs_new_rope)

    # 4. 简单校验
    assert isinstance(mha.rotary_emb, new_rope)
    return mha