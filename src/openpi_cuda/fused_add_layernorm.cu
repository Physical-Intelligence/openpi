#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops.h"

// Warp-level reduction for sum
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// FUSED ADD + LAYERNORM KERNEL
// Combines: output = LayerNorm(x + residual)
// This saves one memory round-trip by avoiding writing the intermediate sum
// ============================================================================

template <int HIDDEN_SIZE>
__global__ void fused_add_layernorm_kernel_f32(
    const float* __restrict__ x,         // [batch, seq_len, hidden_size]
    const float* __restrict__ residual,  // [batch, seq_len, hidden_size]
    const float* __restrict__ gamma,     // [hidden_size]
    const float* __restrict__ beta,      // [hidden_size]
    float* __restrict__ output,          // [batch, seq_len, hidden_size]
    float* __restrict__ sum_output, const int num_tokens, const float eps) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid = threadIdx.x;
    const int offset = token_idx * HIDDEN_SIZE;

    // Shared memory for reduction
    __shared__ float s_mean;
    __shared__ float s_variance;

    // Step 1: Compute sum and mean using vectorized loads
    constexpr int VEC_SIZE = 4;
    constexpr int NUM_VEC = HIDDEN_SIZE / VEC_SIZE;        // 1152/4 = 288
    constexpr int VEC_PER_THREAD = (NUM_VEC + 255) / 256;  // 2

    float thread_sum = 0.0f;

// Each thread computes sum for its assigned elements
#pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; i++) {
        const int vec_idx = tid + i * 256;
        if (vec_idx < NUM_VEC) {
            const int idx = vec_idx * VEC_SIZE;

            // Vectorized load of 4 floats
            float4 x_vec = *reinterpret_cast<const float4*>(&x[offset + idx]);
            float4 res_vec = *reinterpret_cast<const float4*>(&residual[offset + idx]);

            // Add and accumulate for mean
            thread_sum += x_vec.x + res_vec.x;
            thread_sum += x_vec.y + res_vec.y;
            thread_sum += x_vec.z + res_vec.z;
            thread_sum += x_vec.w + res_vec.w;
        }
    }

    // Block-level reduction for mean
    __shared__ float partial_sums[32];
    float warp_sum = warpReduceSum(thread_sum);
    if (tid % 32 == 0) {
        partial_sums[tid / 32] = warp_sum;
    }
    __syncthreads();

    if (tid < 32) {
        float val = tid < 8 ? partial_sums[tid] : 0.0f;
        val = warpReduceSum(val);
        if (tid == 0) {
            s_mean = val / HIDDEN_SIZE;
        }
    }
    __syncthreads();

    // Step 2: Compute variance
    float thread_var = 0.0f;

#pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; i++) {
        const int vec_idx = tid + i * 256;
        if (vec_idx < NUM_VEC) {
            const int idx = vec_idx * VEC_SIZE;

            float4 x_vec = *reinterpret_cast<const float4*>(&x[offset + idx]);
            float4 res_vec = *reinterpret_cast<const float4*>(&residual[offset + idx]);

            // Compute deviations from mean
            float sum0 = x_vec.x + res_vec.x;
            float sum1 = x_vec.y + res_vec.y;
            float sum2 = x_vec.z + res_vec.z;
            float sum3 = x_vec.w + res_vec.w;

            float diff0 = sum0 - s_mean;
            float diff1 = sum1 - s_mean;
            float diff2 = sum2 - s_mean;
            float diff3 = sum3 - s_mean;

            thread_var += diff0 * diff0;
            thread_var += diff1 * diff1;
            thread_var += diff2 * diff2;
            thread_var += diff3 * diff3;
        }
    }

    // Block-level reduction for variance
    float warp_var = warpReduceSum(thread_var);
    if (tid % 32 == 0) {
        partial_sums[tid / 32] = warp_var;
    }
    __syncthreads();

    if (tid < 32) {
        float val = tid < 8 ? partial_sums[tid] : 0.0f;
        val = warpReduceSum(val);
        if (tid == 0) {
            s_variance = val / HIDDEN_SIZE;
        }
    }
    __syncthreads();

    // Step 3: Normalize and apply gamma/beta
    const float inv_std = rsqrtf(s_variance + eps);

#pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; i++) {
        const int vec_idx = tid + i * 256;
        if (vec_idx < NUM_VEC) {
            const int idx = vec_idx * VEC_SIZE;

            float4 x_vec = *reinterpret_cast<const float4*>(&x[offset + idx]);
            float4 res_vec = *reinterpret_cast<const float4*>(&residual[offset + idx]);
            float4 gamma_vec = *reinterpret_cast<const float4*>(&gamma[idx]);
            float4 beta_vec = *reinterpret_cast<const float4*>(&beta[idx]);

            // Normalize and apply affine transform
            float4 out_vec;
            out_vec.x = ((x_vec.x + res_vec.x) - s_mean) * inv_std * gamma_vec.x + beta_vec.x;
            out_vec.y = ((x_vec.y + res_vec.y) - s_mean) * inv_std * gamma_vec.y + beta_vec.y;
            out_vec.z = ((x_vec.z + res_vec.z) - s_mean) * inv_std * gamma_vec.z + beta_vec.z;
            out_vec.w = ((x_vec.w + res_vec.w) - s_mean) * inv_std * gamma_vec.w + beta_vec.w;

            *reinterpret_cast<float4*>(&output[offset + idx]) = out_vec;

            float4 sum_vec;
            sum_vec.x = x_vec.x + res_vec.x;
            sum_vec.y = x_vec.y + res_vec.y;
            sum_vec.z = x_vec.z + res_vec.z;
            sum_vec.w = x_vec.w + res_vec.w;
            *reinterpret_cast<float4*>(&sum_output[offset + idx]) = sum_vec;
        }
    }
}

// Specialized kernel for bfloat16 - converts to float for computation
template <int HIDDEN_SIZE>
__global__ void fused_add_layernorm_kernel_bf16(const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ residual,
                                                const __nv_bfloat16* __restrict__ gamma,
                                                const __nv_bfloat16* __restrict__ beta,
                                                __nv_bfloat16* __restrict__ output,
                                                __nv_bfloat16* __restrict__ sum_output,
                                                const int num_tokens, const float eps) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid = threadIdx.x;
    const int offset = token_idx * HIDDEN_SIZE;

    __shared__ float s_mean;
    __shared__ float s_variance;

    // Process 4 elements per iteration
    constexpr int ELEMS_PER_ITER = 4;
    constexpr int NUM_ITERS = HIDDEN_SIZE / ELEMS_PER_ITER;
    constexpr int ITERS_PER_THREAD = (NUM_ITERS + 255) / 256;

    float thread_sum = 0.0f;

#pragma unroll
    for (int i = 0; i < ITERS_PER_THREAD; i++) {
        const int iter_idx = tid + i * 256;
        if (iter_idx < NUM_ITERS) {
            const int idx = iter_idx * ELEMS_PER_ITER;

#pragma unroll
            for (int j = 0; j < ELEMS_PER_ITER; j++) {
                float x_val = __bfloat162float(x[offset + idx + j]);
                float res_val = __bfloat162float(residual[offset + idx + j]);
                thread_sum += x_val + res_val;
            }
        }
    }

    // Reduce for mean
    __shared__ float partial_sums[32];
    float warp_sum = warpReduceSum(thread_sum);
    if (tid % 32 == 0) {
        partial_sums[tid / 32] = warp_sum;
    }
    __syncthreads();

    if (tid < 32) {
        float val = tid < 8 ? partial_sums[tid] : 0.0f;
        val = warpReduceSum(val);
        if (tid == 0) {
            s_mean = val / HIDDEN_SIZE;
        }
    }
    __syncthreads();

    // Compute variance
    float thread_var = 0.0f;

#pragma unroll
    for (int i = 0; i < ITERS_PER_THREAD; i++) {
        const int iter_idx = tid + i * 256;
        if (iter_idx < NUM_ITERS) {
            const int idx = iter_idx * ELEMS_PER_ITER;

#pragma unroll
            for (int j = 0; j < ELEMS_PER_ITER; j++) {
                float x_val = __bfloat162float(x[offset + idx + j]);
                float res_val = __bfloat162float(residual[offset + idx + j]);
                float sum_val = x_val + res_val;
                float diff = sum_val - s_mean;
                thread_var += diff * diff;
            }
        }
    }

    // Reduce for variance
    float warp_var = warpReduceSum(thread_var);
    if (tid % 32 == 0) {
        partial_sums[tid / 32] = warp_var;
    }
    __syncthreads();

    if (tid < 32) {
        float val = tid < 8 ? partial_sums[tid] : 0.0f;
        val = warpReduceSum(val);
        if (tid == 0) {
            s_variance = val / HIDDEN_SIZE;
        }
    }
    __syncthreads();

    // Normalize
    const float inv_std = rsqrtf(s_variance + eps);

#pragma unroll
    for (int i = 0; i < ITERS_PER_THREAD; i++) {
        const int iter_idx = tid + i * 256;
        if (iter_idx < NUM_ITERS) {
            const int idx = iter_idx * ELEMS_PER_ITER;

#pragma unroll
            for (int j = 0; j < ELEMS_PER_ITER; j++) {
                float x_val = __bfloat162float(x[offset + idx + j]);
                float res_val = __bfloat162float(residual[offset + idx + j]);
                float gamma_val = __bfloat162float(gamma[idx + j]);
                float beta_val = __bfloat162float(beta[idx + j]);

                float sum_val = x_val + res_val;
                float normalized = (sum_val - s_mean) * inv_std;
                float result = normalized * gamma_val + beta_val;

                output[offset + idx + j] = __float2bfloat16(result);

                sum_output[offset + idx + j] = __float2bfloat16(sum_val);
            }
        }
    }
}

// Host function
std::tuple<torch::Tensor, torch::Tensor> fused_add_layernorm_cuda(torch::Tensor x,
                                                                  torch::Tensor residual,
                                                                  torch::Tensor gamma,
                                                                  torch::Tensor beta, float eps) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(residual.is_cuda(), "residual must be on CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be on CUDA");
    TORCH_CHECK(beta.is_cuda(), "beta must be on CUDA");

    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, seq_len, hidden_size]");
    TORCH_CHECK(residual.dim() == 3, "residual must be 3D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D");

    const int batch_size = x.size(0);
    const int seq_len = x.size(1);
    const int hidden_size = x.size(2);

    TORCH_CHECK(residual.size(0) == batch_size && residual.size(1) == seq_len &&
                    residual.size(2) == hidden_size,
                "residual shape must match x");
    TORCH_CHECK(gamma.size(0) == hidden_size, "gamma size must match hidden_size");
    TORCH_CHECK(beta.size(0) == hidden_size, "beta size must match hidden_size");

    // Ensure contiguous
    x = x.contiguous();
    residual = residual.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();

    // Create output
    auto output = torch::empty_like(x);
    auto sum_output = torch::empty_like(x);

    const int num_tokens = batch_size * seq_len;

    // Check if we should use specialized SiGLIP kernel (hidden_size = 1152)
    if (hidden_size == 1152) {
        const int threads = 256;
        const int blocks = num_tokens;

        if (x.scalar_type() == at::ScalarType::Float) {
            fused_add_layernorm_kernel_f32<1152><<<blocks, threads>>>(
                x.data_ptr<float>(), residual.data_ptr<float>(), gamma.data_ptr<float>(),
                beta.data_ptr<float>(), output.data_ptr<float>(), sum_output.data_ptr<float>(),
                num_tokens, eps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (x.scalar_type() == at::ScalarType::BFloat16) {
            fused_add_layernorm_kernel_bf16<1152><<<blocks, threads>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(sum_output.data_ptr<at::BFloat16>()), num_tokens,
                eps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            TORCH_CHECK(false, "Unsupported dtype for specialized SiGLIP kernel");
        }
    } else {
        TORCH_CHECK(false, "Only hidden_size=1152 is supported in specialized kernel");
    }

    return std::make_tuple(output, sum_output);
}
