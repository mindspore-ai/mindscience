/******************************************************************************
 * Copyright (c) 2024, Michael Poli.
 ******************************************************************************/

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "causal_conv1d.h"
#include "causal_conv1d_common.h"

template<typename input_t, typename weight_t>
__global__ void step_fir_fwd_kernel(
    input_t* __restrict__ u,
    input_t* __restrict__ fir_state,
    weight_t* __restrict__ weight,
    weight_t* __restrict__ bias,
    input_t* __restrict__ out,
    input_t* __restrict__ out_state,
    const int batch,
    const int dim,
    const int filter_length,
    const int cache_size,
    const bool gated_bias,
    const bool flip_filter
) {
    const int b = blockIdx.x;
    const int d = blockIdx.y;
    const int tid = threadIdx.x;

    if (b >= batch || d >= dim) return;

    // Load inputs
    float u_val = (float)u[b * dim + d];
    float state_vals[128];  // Assume max filter length of 128
    
    #pragma unroll
    for (int i = 0; i < cache_size; i++) {
        state_vals[i] = (float)fir_state[b * dim * cache_size + d * cache_size + i];
    }

    // Load and process weight
    float weight_vals[128];
    if (flip_filter) {
        #pragma unroll
        for (int i = 0; i <= cache_size; i++) {
            weight_vals[i] = (float)weight[d * filter_length + filter_length - 1 - i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i <= cache_size; i++) {
            weight_vals[i] = (float)weight[d * filter_length + i];
        }
    }

    // Compute output
    float h0 = weight_vals[cache_size];
    float y = h0 * u_val;
    
    #pragma unroll
    for (int i = 0; i < cache_size; i++) {
        y += state_vals[i] * weight_vals[i];
    }

    // Add bias if present
    if (bias != nullptr) {
        float bias_val = (float)bias[d];
        if (gated_bias) {
            y += bias_val * u_val;
        } else {
            y += bias_val;
        }
    }

    // Update state
    if (cache_size < filter_length - 1) {
        #pragma unroll
        for (int i = 0; i < cache_size; i++) {
            out_state[b * dim * (cache_size + 1) + d * (cache_size + 1) + i] = fir_state[b * dim * cache_size + d * cache_size + i];
        }
        out_state[b * dim * (cache_size + 1) + d * (cache_size + 1) + cache_size] = u_val;
    } else {
        #pragma unroll
        for (int i = 0; i < cache_size - 1; i++) {
            out_state[b * dim * cache_size + d * cache_size + i] = state_vals[i + 1];
        }
        out_state[b * dim * cache_size + d * cache_size + cache_size - 1] = u_val;
    }

    // Write output
    out[b * dim + d] = (input_t)y;
}

template<typename input_t, typename weight_t>
void step_fir_fwd_cuda(
    input_t* u,
    input_t* fir_state,
    weight_t* weight,
    weight_t* bias,
    input_t* out,
    input_t* out_state,
    const int batch,
    const int dim,
    const int filter_length,
    const int cache_size,
    const bool gated_bias,
    const bool flip_filter,
    cudaStream_t stream
) {
    dim3 grid(batch, dim);
    dim3 block(128);

    step_fir_fwd_kernel<input_t, weight_t><<<grid, block, 0, stream>>>(
        u, fir_state, weight, bias, out, out_state,
        batch, dim, filter_length, cache_size,
        gated_bias, flip_filter
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Explicit instantiations
template void step_fir_fwd_cuda<float, float>(float*, float*, float*, float*, float*, float*, int, int, int, int, bool, bool, cudaStream_t);
template void step_fir_fwd_cuda<at::Half, float>(at::Half*, at::Half*, float*, float*, at::Half*, at::Half*, int, int, int, int, bool, bool, cudaStream_t);
template void step_fir_fwd_cuda<at::BFloat16, float>(at::BFloat16*, at::BFloat16*, float*, float*, at::BFloat16*, at::BFloat16*, int, int, int, int, bool, bool, cudaStream_t);