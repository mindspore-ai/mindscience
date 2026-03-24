# Copyright (c) 2024, Michael Poli.

import gc

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint, ops
from mindscience.sciops import fft
from .utils import trim_pad_tensor

try:
    pass
except:
    pass
from vortex.model.utils import column_split
from vortex.logging import activations_logger

IIR_PREFILL_MODES = [
    "recurrence",
    "modal-fft",
    "hybrid-modal-recurrence",
    "modal-scan",
    "canonical-fft",
    "iir-fir-caching",
]


def adjust_filter_shape_for_broadcast(u, h_real, h_imag):
    h_real = h_real.squeeze()  # Standardize to [D, L] from [1, D, L] and [D, 1, L]
    h_imag = h_imag.squeeze()  # Standardize to [D, L] from [1, D, L] and [D, 1, L]

    # Case: u: [B, D, L], k_f: [D, L]
    if len(u.shape) > len(h_real.shape):
        h_real = h_real.unsqueeze(0)
        h_imag = h_imag.unsqueeze(0)

    # Case: u: [B, D1, D2, L], k_f: [B, D, L]
    if len(u.shape) > 3:
        h_real = h_real.unsqueeze(1)
        h_imag = h_imag.unsqueeze(1)
    return h_real, h_imag

def complex_mul(r1, i1, r2, i2):
    return r1*r2 - i1*i2,  r1*i2 + i1*r2

def fftconv_func(
    u,
    k,
    D,
    dropout_mask,
    gelu=True,
    k_rev=None,
    bidirectional=False,
    print_activations=False,
    layer_idx=None,
    **kwargs,
):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f_real, k_f_imag = fft.asd_rfftn(trim_pad_tensor(k, fft_size))
    k_f_real, k_f_imag = k_f_real / fft_size, k_f_imag / fft_size
    k_f_real, k_f_imag = adjust_filter_shape_for_broadcast(u, k_f_real, k_f_imag)
    k = k.squeeze()

    u_f_real, u_f_imag = fft.asd_rfftn(trim_pad_tensor(u, fft_size))

    if bidirectional:
        k, k2 = k.split(k.shape[1] // 2, dim=1)
        k2_f_real, k2_f_imag = fft.asd_rfftn(k2) / fft_size
        y1_real, y1_imag = complex_mul(u_f_real, u_f_imag, k_f_real, k_f_imag)
        y2_real, y2_imag = complex_mul(u_f_real, -u_f_imag, k2_f_real, -k2_f_imag)

        y_real = y1_real + y2_real
        y_imag = y1_imag + y2_imag
        y = fft.asd_irfftn(y_real, y_imag, n=fft_size)[..., :seqlen]

    else:
        if k_rev is not None:
            k_rev_f_r, k_rev_f_i = fft.asd_rfftn(trim_pad_tensor(k_rev, fft_size))
            k_rev_real, k_rev_imag = k_rev_f_r / fft_size, k_rev_f_i / fft_size
            k_f_real = k_f_real + k_rev_real
            k_f_imag = k_f_imag - k_rev_imag
        u_k_real, u_k_imag = complex_mul(u_f_real, u_f_imag, k_f_real, k_f_imag)
        y = fft.asd_irfftn(u_k_real, u_k_imag, n=fft_size)[..., :seqlen]

    if print_activations:
        activations_logger.info(f"post fftconv pre bias {y} {y.min()} {y.max()}")

    out = y + u * D.unsqueeze(-1)

    if print_activations:
        activations_logger.info(f"post fftconv post bias {out} {out.min()} {out.max()}")

    return out.astype(dtype=u.dtype)


def canonicalize_modal_system(poles, residues):
    """Canonicalize a modal system.

    Args:
        poles (Tensor): The poles of the system.
        residues (Tensor): The residues of the system.

    Returns:
        Tuple[Tensor, Tensor]: The canonicalized poles and residues.
    """
    raise NotImplementedError


def list_tensors(idx):
    for obj in gc.get_objects():
        try:
            if ms.is_tensor(obj) and isinstance(obj, ms.Tensor):
                # dump to log
                print(type(obj), obj.size())
                el = obj[0]
                with open(f"tensors_{idx}.txt", "a") as f:
                    f.write(f"{type(obj)} {obj.size()} {el}\n")
        except Exception:
            pass


class HyenaInferenceEngine:
    def __init__(
        self,
        fir_fn=None,
        iir_prefill_style="modal-fft",
        layer_idx=None,
        ground_truth_activations_path=None,
        print_activations=False,
        hyena_flip_x1x2=False,
    ) -> None:
        self.fir_fn = fir_fn
        assert iir_prefill_style in IIR_PREFILL_MODES, f"iir_prefill_style must be one of {IIR_PREFILL_MODES}"
        self.iir_prefill_style = iir_prefill_style
        self.layer_idx = layer_idx
        self.low_mem_mode = False
        self.ground_truth_activations_path = ground_truth_activations_path
        self.print_activations = print_activations
        self.hyena_flip_x1x2 = hyena_flip_x1x2

    def parallel_fir(
        self,
        fir_fn,
        u,
        weight,
        bias,
        L,
        dims,
        groups=None,
        gated_bias=False,
        column_split_hyena=False,
        dim_last=True,
        fir_length=3,
        gate=False,
        inference_params=None,
        prefill_mode=None,
        padding_mask=None,
    ):
        L = u.shape[1] if dim_last else u.shape[2]
        if gate:
            hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
            # Compatibility with training infra that column splits the projections
            if column_split_hyena:
                x2, x1, v = column_split(u, num_attention_heads, hidden_size_per_attention_head)
            else:
                x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
            if self.hyena_flip_x1x2:
                x1, x2 = x2, x1
            u = x1 * v

            if self.print_activations:
                activations_logger.info(f"q: {x2}, {x2.min()}, {x2.max()}")
                activations_logger.info(f"k: {x1}, {x1.min()}, {x1.max()}")
                activations_logger.info(f"v: {v}, {v.min()}, {v.max()}")
                activations_logger.info(f"post pregate: {u}, {u.min()}, {u.max()}")

        # prepare input layout, dimensions and dispatch to fir kernel
        # Deprecated
        if fir_fn != F.conv1d:
            if dim_last:
                u = u.permute(0, 2, 1)  # B, D, L
            z = fir_fn(u)[:, :L]  # B, L, D

        elif fir_length >= 128:
            u_fp32 = u.astype(ms.float32)
            weight_fp32 = weight[:, :, :L].astype(ms.float32)
            z = fftconv_func(
                u_fp32,
                weight_fp32,
                bias,
                None,
                gelu=False,
                bidirectional=False,
                print_activations=self.print_activations,
                groups=groups,
                layer_idx=self.layer_idx,
            )
            z = z.astype(u.dtype)
        else:
            if dim_last:
                u = u.permute(0, 2, 1)  # B, D, L

            if groups is None:
                g = u.shape[1]
            else:
                g = groups

            u_fp32 = u.astype(ms.float32)
            weight_fp32 = weight.astype(ms.float32)
            z = fir_fn(
                u_fp32,
                weight_fp32,
                bias=None,
                stride=1,
                padding=fir_length - 1,
                groups=u.shape[1],  # always set to D, regardless of filter grouping
            )[..., :L]
            if self.print_activations:
                activations_logger.info(f"post filter: {z}, {z.min()}, {z.max()}")

            z = z.astype(u.dtype)

            if gated_bias is False:
                if self.print_activations:
                    activations_logger.info(f"post dw conv {z} {z.min()} {z.max()}")
                    # if self.ground_truth_activations_path:
                    #     z_savanna = torch.load(f"{self.ground_truth_activations_path}/post_dw_conv_{self.layer_idx}.pt")
                    #     z_savanna = z_savanna.permute(1, 2, 0)
                    #     z_diff = (z.squeeze() - z_savanna.squeeze()).abs().max()
                    #     activations_logger.info(f"dw_conv_diff: {z_diff}")

            if bias is not None:
                if gated_bias:
                    z = z + bias[None, :, None] * u
                else:
                    z = z + bias[None, :, None]

        # handle padding post fir, the only place with biases
        if type(padding_mask) == ms.Tensor:
            z = z * padding_mask[:, None]

        if gate:
            # if self.layer_idx == 1:
            #    breakpoint()
            z = x2 * z

            if self.print_activations:
                activations_logger.info(f"hyena filter: {weight}, {weight.min()}, {weight.max()}")
                activations_logger.info(f"post postgate: {z}, {z.min()}, {z.max()}")
                # if self.ground_truth_activations_path:
                #     q_savanna = torch.load(f"{self.ground_truth_activations_path}/q_{self.layer_idx}.pt")
                #     k_savanna = torch.load(f"{self.ground_truth_activations_path}/k_{self.layer_idx}.pt")
                #     v_savanna = torch.load(f"{self.ground_truth_activations_path}/v_{self.layer_idx}.pt")

                #     q_diff = (x2 - q_savanna).abs()
                #     k_diff = (x1 - k_savanna).abs()
                #     v_diff = (v - v_savanna).abs()

                #     activations_logger.info(f"q_diff: {q_diff.max()}, {q_diff.mean()}")
                #     activations_logger.info(f"k_diff: {k_diff.max()}, {k_diff.mean()}")
                #     activations_logger.info(f"v_diff: {v_diff.max()}, {v_diff.mean()}")

                #     h_savanna = torch.load(f"/home/zymrael/checkpoints/evo2/activations/savanna/hyena_filter_{self.layer_idx}.pt")
                #     h_diff = (weight[..., :h_savanna.shape[-1]].squeeze() - h_savanna.squeeze()).abs()

                #     activations_logger.info(f"h_diff: {h_diff.max()}, {h_diff.mean()}")

        if inference_params is not None:
            fir_state = u[..., -fir_length + 1 :]
        else:
            fir_state = None

        return z, fir_state

    def parallel_iir(
        self,
        z_pre,
        h,
        D,
        L,
        poles,
        residues,
        t,
        dims,
        layer_idx,
        inference_params=None,
        prefill_style="fft",
        fftconv_fn=None,
        padding_mask=None,
        use_flashfft=False,
        column_split_hyena=False,
        long_fir_threshold=None,
    ):
        """Compute the output state of the short convolutional filter."""
        fft_size = 2 * L
        hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
        # Compatibility with training infra that column splits the projections
        if column_split_hyena:
            z = z_pre.reshape(
                z_pre.shape[0],
                num_attention_heads,
                3 * hidden_size_per_attention_head,
                z_pre.shape[2],
            )
            x2, x1, v = (
                z[:, :, :hidden_size_per_attention_head],
                z[
                    :,
                    :,
                    hidden_size_per_attention_head : 2 * hidden_size_per_attention_head,
                ],
                z[:, :, 2 * hidden_size_per_attention_head :],
            )
            x2, x1, v = (
                x2.reshape(x2.shape[0], -1, x2.shape[-1]),
                x1.reshape(x1.shape[0], -1, x1.shape[-1]),
                v.reshape(v.shape[0], -1, v.shape[-1]),
            )
        else:
            x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)

        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        x1v = x1 * v

        if inference_params is not None and prefill_style == "recurrence":
            y = self.prefill_via_direct_recurrence(
                inference_params=inference_params,
                x1v=x1v,
                L=L,
                poles=poles,
                residues=residues,
            )

        else:
            if use_flashfft and (L % 2) == 0:  # only works with even L
                y = fftconv_fn(
                    x1v.astype(dtype=ms.bfloat16).contiguous(),
                    h.astype(dtype=ms.float32),
                )
                X_s = None

            elif long_fir_threshold is None:
                h_fp32 = h.astype(dtype=ms.float32)
                H_real, H_imag = fft.asd_rfftn(trim_pad_tensor(h_fp32, fft_size))
                H_real, H_imag = H_real / fft_size, H_imag / fft_size
                # H = ops.Complex()(H_real, H_imag)
                x1v_fp32 = x1v.astype(dtype=ms.float32)
                x1v_trim = trim_pad_tensor(x1v_fp32, fft_size)
                X_s_real, X_s_imag = fft.asd_fftn(x1v_trim, ops.zeros(x1v_trim.shape, dtype=ms.float32))
                X_real = X_s_real[..., : H_real.shape[-1]]
                X_imag = X_s_imag[..., : H_imag.shape[-1]]
                if len(z_pre.shape) > 3:
                    # H = H.unsqueeze(1)
                    H_real = H_real.unsqueeze(1)
                    H_imag = H_imag.unsqueeze(1)
                # x_h = X * H
                x_h_r, x_h_i = complex_mul(X_real, X_imag, H_real, H_imag)
                y = fft.asd_irfftn(x_h_r, x_h_i, n=fft_size)[..., :L]

            else:
                assert h.shape[0] == 1, "batch size must be 1 for long_fir_threshold"
                h = h[0][:, None]  # rearrange to d, 1, l for depthwise conv1d
                h = h[..., :long_fir_threshold]
                y = F.conv1d(
                    x1v,
                    h.astype(dtype=x1v.dtype),
                    stride=1,
                    groups=x1v.shape[1],
                    padding=h.shape[-1] - 1,
                )[..., :L]
        y = y.astype(dtype=x1v.dtype)
        y = (y + x1v * D.unsqueeze(-1)) * x2

        if self.print_activations:
            activations_logger.info(f"hyena filter: {h}, {h.min()}, {h.max()}")
            activations_logger.info(f"post hyena iir gate: {y}, {y.min()}, {y.max()}")
            activations_logger.info(f"q: {x2}, {x2.min()}, {x2.max()}")
            activations_logger.info(f"k: {x1}, {x1.min()}, {x1.max()}")
            activations_logger.info(f"v: {v}, {v.min()}, {v.max()}")
            # if self.ground_truth_activations_path:
            #     q_savanna = torch.load(f"{self.ground_truth_activations_path}/q_{self.layer_idx}.pt")
            #     k_savanna = torch.load(f"{self.ground_truth_activations_path}/k_{self.layer_idx}.pt")
            #     v_savanna = torch.load(f"{self.ground_truth_activations_path}/v_{self.layer_idx}.pt")

            #     q_diff = (x2 - q_savanna).abs()
            #     k_diff = (x1 - k_savanna).abs()
            #     v_diff = (v - v_savanna).abs()

            #     activations_logger.info(f"q_diff: {q_diff.max()}, {q_diff.mean()}")
            #     activations_logger.info(f"k_diff: {k_diff.max()}, {k_diff.mean()}")
            #     activations_logger.info(f"v_diff: {v_diff.max()}, {v_diff.mean()}")

            #     h_savanna = torch.load(f"/home/zymrael/checkpoints/evo2/activations/savanna/hyena_filter_{self.layer_idx}.pt")

            #     h_diff = (h[..., :h_savanna.shape[-1]].squeeze() - h_savanna.squeeze()).abs()
            #     activations_logger.info(f"h_diff: {h_diff.max()}, {h_diff.mean()}")

        if inference_params is not None:
            if prefill_style == "fft":
                self.prefill_via_modal_fft(
                    inference_params=inference_params,
                    x1v=x1v,
                    X_s_real=X_s_real,
                    X_s_imag=X_s_imag,
                    L=L,
                    t=t,
                    poles=poles,
                    dims=dims,
                    layer_idx=layer_idx,
                    use_flashfft=use_flashfft,
                    fftconv_fn=fftconv_fn,
                )

            elif prefill_style == "recurrence":
                # recurrent prefill is done before
                pass
            else:
                raise NotImplementedError
            if self.low_mem_mode:
                # TODO: smarter gc
                del z_pre, x2, x1, v, x1v, h, poles, residues
                ms.runtime.empty_cache()

        return y.permute(0, 2, 1)

    def step_fir(self, u, fir_state, weight, bias=None, gated_bias=False, flip_filter=False):
        """Steps forward FIR filters in the architecture.

        FIR filters generally include truncated convolutions in Hyena with an explicit or hybrid time-domain parametrization:
        * Short FIR filters in Hyena featurizers
        * Short and medium FIR filters in Hyena operators

        Note:
            `fir_state` contains the last FIR filter length - 1 elements of `u`: `u_(L-2), u_{L-1), ...`
            We assume dimensions of `short_filter_weight` to be `[d, 1, short_filter_len]`.
        """
        weight = weight.squeeze()

        cache_size = fir_state.shape[-1]
        filter_length = weight.shape[-1]
        if flip_filter:
            weight = weight.flip(-1)
            weight = weight[..., -cache_size - 1 :].unsqueeze(0)
        else:
            weight = weight[..., : cache_size + 1].unsqueeze(0)

        input_dtype = u.dtype
        weight = weight.astype(ms.float32)
        u = u.astype(ms.float32)
        fir_state = fir_state.astype(ms.float32)
        bias = bias.astype(ms.float32) if bias is not None else None

        h0, h = weight[..., -1], weight[..., :-1]
        y = h0 * u + mint.sum(fir_state * h, dim=-1)

        if bias is not None:
            if gated_bias:
                y = y + bias * u
            else:
                y = y + bias

        # Update the state
        if cache_size < filter_length - 1:
            fir_state = mint.cat([fir_state, u[..., None]], dim=-1)
        else:
            fir_state = mint.roll(fir_state, -1, dims=2)
            fir_state[..., -1] = u

        return y.astype(input_dtype), fir_state

    def step_iir(self, x2, x1, v, D, residues, poles, iir_state, iir_groups=1):
        # TODO: kernelize
        x1v = x1 * v
        poles = mint.exp(poles)  # poles arg contains log_poles
        poles = poles[..., 0][None]  # squeeze dummy seqlen dim and add dummy batch dim
        residues = residues[None]  # add dummy batch dim
        iir_state = poles * iir_state + x1v[..., None]

        res_state = mint.sum(residues * iir_state, dim=-1)

        if iir_groups > 1:
            raise NotImplementedError
        # if self.layer_idx == 2:
        #    breakpoint()
        y = x2 * (res_state + D * x1v)

        return y, iir_state

    def prefill_via_fir_caching(self, u, inference_params, L, *args, **kwargs):
        """Turns the IIR filter into a FIR and uses a cache for decoding."""
        raise NotImplementedError(":)")

    def prefill_via_direct_recurrence(self, inference_params, x1v, L, residues, poles, *args, **kwargs) -> ms.Tensor:
        """
        Compute the IIR state via explicit recurrence (modal form)

        This is the most memory efficient prefilling method for Hyena filters.

        Note:
            dtypes: [state: float32, poles: float32, x1v: bfloat16, output: bfloat16]
        """
        state_dim = poles.shape[1]
        x1v_ = x1v[..., None, None]  # b, d, l, sdim, reim
        x1v_ = x1v_.repeat(1, 1, 1, state_dim, 2)  # b, d, l, sdim, reim
        x1v_[..., 1] = 0

        state = 0 * x1v_[:, :, 0]
        output = 0 * x1v_[:, :, :, 0, 0]  # b, d, l

        # suppress dummy seqlen dimension
        poles = poles[:, :, 0][None]
        residues = residues[:, :, 0][None].repeat(x1v_.shape[0], 1, 1, 1)  # b, d, sdim, reim

        # state: b, d, sdim, reim
        # poles: 1, d, sdim, reim
        # x1v_: b, d, l, sdim, reim
        for i in range(L):
            state[..., 0] = poles[..., 0] * state[..., 0] - poles[..., 1] * state[..., 1] + x1v_[:, :, i, :, 0]
            state[..., 1] = poles[..., 0] * state[..., 1] + poles[..., 1] * state[..., 0] + x1v_[:, :, i, :, 1]
            output[:, :, i] = mint.sum(residues * state, dim=-2)[..., 0]  # .real

        inference_params.state_dict[self.layer_idx] = state.astype(dtype=ms.float32)

        return output

    def prefill_via_hybrid_recurrence(self, inference_params, u, log_poles, x1v_f_a, L, *args, **kwargs):
        """
        Compute the IIR state via hybrid recurrence-convolution over blocks
        """
        raise NotImplementedError(":)")

    def prefill_via_scan(self, u, inference_params=None, *args, **kwargs):
        raise NotImplementedError

    def prefill_via_canonical_fft(self, u, inference_params=None, *args, **kwargs):
        """
        Compute the IIR state via a single FFT

        This is the most memory efficient "parallelized" prefilling method for Hyena.

        From: https://arxiv.org/abs/2310.18780
        """
        raise NotImplementedError(":)")

    def prefill_via_modal_fft(
        self,
        inference_params,
        x1v,
        L,
        poles,
        t,
        dims,
        layer_idx,
        X_s_real=None,
        X_s_imag=None,
        use_flashfft=False,
        fftconv_fn=None,
        state_dtype=ms.float32,
        *args,
        **kwargs,
    ):
        """
        Compute the IIR state via a single FFT
        """
        # When the model has a long convolution derived from a recurrence in modal form and prefill_style is "fft",
        # we split the filter into poles and residues and reuse FFT computation on the input.
        hidden_size, _, _, state_size, hyena_filter_groups = dims

        bs = x1v.shape[0]
        fft_size = 2 * L
        # poles = torch.view_as_complex(poles.to(torch.float32))
        state_s = (poles.astype(ms.float32) * t).exp()

        # state_s = poles**t
        state_s_fft = trim_pad_tensor(state_s, fft_size)
        state_S_real, state_S_imag = fft.asd_fftn(
            state_s_fft,
            ops.zeros(state_s_fft.shape, dtype=ms.float32),
        )
        state_S_real = state_S_real.repeat(bs, 1, 1, 1)  # B, D, state_dim, 2 * L
        state_S_imag = state_S_imag.repeat(bs, 1, 1, 1)  # B, D, state_dim, 2 * L
        if hyena_filter_groups > 1:
            state_S_real = state_S_real.repeat_interleave(hidden_size // hyena_filter_groups, 1)
            state_S_imag = state_S_imag.repeat_interleave(hidden_size // hyena_filter_groups, 1)
        X_s_real = X_s_real[..., None, :]
        X_s_imag = X_s_imag[..., None, :]
        pre_state_real, pre_state_imag = complex_mul(X_s_real, X_s_imag, state_S_real, state_S_real)
        state_real, state_imag = fft.asd_ifftn(pre_state_real, pre_state_imag)
        assert state_real.shape[-1] == fft_size
        state = ops.Complex()(state_real, state_imag)
        inference_params.state_dict[layer_idx] = state[..., L - 1].astype(dtype=state_dtype)


class HyenaFilter:
    """Handles Hyena filter computations including FFT and direct convolution."""

    def __init__(self, use_flash_fft=False):
        self.use_flash_fft = use_flash_fft

    def fft_conv(self, u, k, D, **kwargs):
        """FFT-based convolution implementation."""
        seqlen = u.shape[-1]
        fft_size = 2 * seqlen

        k_f = self._prepare_filter(k, u, fft_size)
        y = self._compute_fft_conv(u, k_f, fft_size, seqlen, **kwargs)

        return y + u * D.unsqueeze(-1)

    def _prepare_filter(self, k, u, fft_size):
        """Prepare filter for FFT convolution."""
        k_f_real, k_f_imag = fft.asd_rfftn(trim_pad_tensor(k, fft_size))
        k_f = ops.Complex(k_f_real, k_f_imag) / fft_size
        return adjust_filter_shape_for_broadcast(u, k_f)
