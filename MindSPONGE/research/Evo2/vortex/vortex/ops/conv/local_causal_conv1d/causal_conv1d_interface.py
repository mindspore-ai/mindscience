# Copyright (c) 2024, Tri Dao.

import torch
import torch.nn.functional as F
from typing import Optional
from torch.library import custom_op
import local_causal_conv1d_cuda


class CausalConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        seq_idx=None,
        initial_states=None,
        return_final_states=False,
        final_states_out=None,
        activation=None,
    ):
        if x.stride(2) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None

        out = local_causal_conv1d_cuda.causal_conv1d_fwd(
            x, weight, bias, seq_idx, initial_states, final_states_out, activation
        )
        ctx.save_for_backward(x, weight, bias, seq_idx, initial_states, activation)
        ctx.return_final_states = return_final_states
        ctx.return_dinitial_states = initial_states is not None and initial_states.requires_grad
        return out if not return_final_states else (out, final_states_out)

    @staticmethod
    def backward(ctx, dout, *args):
        x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        if dout.stride(2) != 1 and dout.stride(1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        # Here we just pass in None and dx will be allocated in the C++ code.
        dx, dweight, dbias, dinitial_states = local_causal_conv1d_cuda.causal_conv1d_bwd(
            x,
            weight,
            bias,
            dout,
            seq_idx,
            initial_states,
            dfinal_states,
            None,
            ctx.return_dinitial_states,
            ctx.activation,
        )
        return (
            dx,
            dweight,
            dbias if bias is not None else None,
            None,
            dinitial_states if initial_states is not None else None,
            None,
            None,
            None,
        )


def old_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    return CausalConv1dFn.apply(x, weight, bias, seq_idx, initial_states, final_states_out, activation)


@custom_op(
    "vortex::causal_conv1d_fwd",
    mutates_args=(),
    device_types=["cuda"],
)
def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    x: (b, d, l)
    weight: (d, hl)
    bias: (d,)
    seq_idx: (b, l)
    initial_states: (b, d, hl - 1)
    final_states_out: (b, d, hl - 1)
    out: (b, d, l)
    """
    if x.stride(2) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    # out = local_causal_conv1d_cuda.causal_conv1d_fwd(
    #    x, weight, bias, seq_idx, initial_states, final_states_out, activation
    # )
    return x.clone()


@causal_conv1d_fn.register_fake
def causal_conv1d_fn_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    return x


def setup_context(ctx, inputs, output) -> torch.Tensor:
    x, weight, bias, seq_idx, initial_states, final_states_out, activation = inputs
    ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)


def backward(ctx, dout, *args):
    x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
    # if dout.stride(2) != 1 and dout.stride(1) != 1:
    #     dout = dout.contiguous()
    # dx, dweight, dbias, dinitial_states = local_causal_conv1d_cuda.causal_conv1d_bwd(
    #     x,
    #     weight,
    #     bias,
    #     dout,
    #     seq_idx,
    #     initial_states,
    #     None,
    #     None,
    #     False,
    #     None,
    # )
    # return (
    #     dx,
    #     dweight,
    #     dbias if bias is not None else None,
    #     None,
    #     None,
    #     None,
    #     None,
    # )
    return (
        dout,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    causal_conv1d_fn,
    backward,
    setup_context=setup_context,
)


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (b, d, l)
    weight: (d, hl)
    bias: (d,)
    seq_idx: (b, l)
    initial_states: (b, d, hl - 1)
    final_states_out: (b, d, hl - 1)
    out: (b, d, l)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


# def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None, conv_state_indices=None):
#     """
#     x: (batch, dim) or (batch, dim, seqlen)
#     conv_state: (batch, dim, state_len), where state_len >= width - 1
#     weight: (dim, width)
#     bias: (dim,)
#     cache_seqlens: (batch,), dtype int32.
#         If not None, the conv_state is treated as a circular buffer.
#         The conv_state will be updated by copying x to the conv_state starting at the index
#         @cache_seqlens % state_len.
#     conv_state_indices: (batch,), dtype int32
#         If None, the conv_state is a larger tensor along the batch dim,
#         and we are selecting the batch coords specified by conv_state_indices.
#         Useful for a continuous batching scenario.

#     out: (batch, dim) or (batch, dim, seqlen)
#     """
#     if activation not in [None, "silu", "swish"]:
#         raise NotImplementedError("activation must be None, silu, or swish")
#     activation = activation in ["silu", "swish"]
#     unsqueeze = x.dim() == 2
#     if unsqueeze:
#         x = x.unsqueeze(-1)
#     out = local_causal_conv1d_cuda.causal_conv1d_update(
#         x, conv_state, weight, bias, activation, cache_seqlens, conv_state_indices
#     )
#     if unsqueeze:
#         out = out.squeeze(-1)
#     return out


# def causal_conv1d_update_ref(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
#     """
#     x: (batch, dim) or (batch, dim, seqlen)
#     conv_state: (batch, dim, state_len), where state_len >= width - 1
#     weight: (dim, width)
#     bias: (dim,)
#     cache_seqlens: (batch,), dtype int32.
#         If not None, the conv_state is treated as a circular buffer.
#         The conv_state will be updated by copying x to the conv_state starting at the index
#         @cache_seqlens % state_len before performing the convolution.

#     out: (batch, dim) or (batch, dim, seqlen)
#     """
#     if activation not in [None, "silu", "swish"]:
#         raise NotImplementedError("activation must be None, silu, or swish")
#     dtype_in = x.dtype
#     unsqueeze = x.dim() == 2
#     if unsqueeze:
#         x = x.unsqueeze(-1)
#     batch, dim, seqlen = x.shape
#     width = weight.shape[1]
#     state_len = conv_state.shape[-1]
#     assert conv_state.shape == (batch, dim, state_len)
#     assert weight.shape == (dim, width)
#     if cache_seqlens is None:
#         x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)  # (batch, dim, state_len + seqlen)
#         conv_state.copy_(x_new[:, :, -state_len:])
#     else:
#         width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
#         width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
#         x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
#         copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
#         copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
#         conv_state.scatter_(2, copy_idx, x)
#     out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
#     if unsqueeze:
#         out = out.squeeze(-1)
#     return (out if activation is None else F.silu(out)).to(dtype=dtype_in)
