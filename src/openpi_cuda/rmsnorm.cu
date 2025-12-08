#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops.h"

// ============================================================================
// RMSNORM KERNEL
// RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * scale
// Simpler than LayerNorm - no mean subtraction, only RMS normalization
// ============================================================================

// Warp-level reduction for sum
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// OPTIMIZED RMSNORM KERNEL (Float32)
// Template parameter for compile-time optimization
// ============================================================================

template <int HIDDEN_SIZE>
__global__ void rmsnorm_kernel_f32(const float* __restrict__ input,
                                   const float* __restrict__ weight,
                                   float* __restrict__ output,
                                   const int num_tokens,
                                   const float eps) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid = threadIdx.x;
    const int offset = token_idx * HIDDEN_SIZE;

    __shared__ float s_variance;

    // Step 1: Compute sum of squares using vectorized loads
    constexpr int VEC_SIZE = 4;
    constexpr int NUM_VEC = HIDDEN_SIZE / VEC_SIZE;
    constexpr int VEC_PER_THREAD = (NUM_VEC + 255) / 256;  // Assuming 256 threads

    float thread_sum = 0.0f;

#pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; i++) {
        const int vec_idx = tid + i * 256;
        if (vec_idx < NUM_VEC) {
            const int idx = vec_idx * VEC_SIZE;

            // Vectorized load of 4 floats
            float4 in_vec = *reinterpret_cast<const float4*>(&input[offset + idx]);

            // Accumulate squared values
            thread_sum += in_vec.x * in_vec.x;
            thread_sum += in_vec.y * in_vec.y;
            thread_sum += in_vec.z * in_vec.z;
            thread_sum += in_vec.w * in_vec.w;
        }
    }

    // Block-level reduction for variance
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
            s_variance = val / HIDDEN_SIZE;  // mean(x^2)
        }
    }
    __syncthreads();

    // Step 2: Normalize and apply scale
    const float inv_rms = rsqrtf(s_variance + eps);

#pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; i++) {
        const int vec_idx = tid + i * 256;
        if (vec_idx < NUM_VEC) {
            const int idx = vec_idx * VEC_SIZE;

            float4 in_vec = *reinterpret_cast<const float4*>(&input[offset + idx]);
            float4 weight_vec = *reinterpret_cast<const float4*>(&weight[idx]);

            // Apply RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + weight)
            float4 out_vec;
            out_vec.x = in_vec.x * inv_rms * (1.0f + weight_vec.x);
            out_vec.y = in_vec.y * inv_rms * (1.0f + weight_vec.y);
            out_vec.z = in_vec.z * inv_rms * (1.0f + weight_vec.z);
            out_vec.w = in_vec.w * inv_rms * (1.0f + weight_vec.w);

            *reinterpret_cast<float4*>(&output[offset + idx]) = out_vec;
        }
    }
}

// ============================================================================
// OPTIMIZED RMSNORM KERNEL (BFloat16)
// ============================================================================

template <int HIDDEN_SIZE>
__global__ void rmsnorm_kernel_bf16(const __nv_bfloat16* __restrict__ input,
                                    const __nv_bfloat16* __restrict__ weight,
                                    __nv_bfloat16* __restrict__ output,
                                    const int num_tokens,
                                    const float eps) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid = threadIdx.x;
    const int offset = token_idx * HIDDEN_SIZE;

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
                float val = __bfloat162float(input[offset + idx + j]);
                thread_sum += val * val;
            }
        }
    }

    // Block-level reduction
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
            s_variance = val / HIDDEN_SIZE;
        }
    }
    __syncthreads();

    // Normalize
    const float inv_rms = rsqrtf(s_variance + eps);

#pragma unroll
    for (int i = 0; i < ITERS_PER_THREAD; i++) {
        const int iter_idx = tid + i * 256;
        if (iter_idx < NUM_ITERS) {
            const int idx = iter_idx * ELEMS_PER_ITER;

#pragma unroll
            for (int j = 0; j < ELEMS_PER_ITER; j++) {
                float in_val = __bfloat162float(input[offset + idx + j]);
                float weight_val = __bfloat162float(weight[idx + j]);
                float result = in_val * inv_rms * (1.0f + weight_val);
                output[offset + idx + j] = __float2bfloat16(result);
            }
        }
    }
}

// ============================================================================
// OPTIMIZED RMSNORM KERNEL (Float16)
// ============================================================================

template <int HIDDEN_SIZE>
__global__ void rmsnorm_kernel_f16(const __half* __restrict__ input,
                                   const __half* __restrict__ weight,
                                   __half* __restrict__ output,
                                   const int num_tokens,
                                   const float eps) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid = threadIdx.x;
    const int offset = token_idx * HIDDEN_SIZE;

    __shared__ float s_variance;

    // Process 8 elements per iteration
    constexpr int ELEMS_PER_ITER = 8;
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
                float val = __half2float(input[offset + idx + j]);
                thread_sum += val * val;
            }
        }
    }

    // Block-level reduction
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
            s_variance = val / HIDDEN_SIZE;
        }
    }
    __syncthreads();

    // Normalize
    const float inv_rms = rsqrtf(s_variance + eps);

#pragma unroll
    for (int i = 0; i < ITERS_PER_THREAD; i++) {
        const int iter_idx = tid + i * 256;
        if (iter_idx < NUM_ITERS) {
            const int idx = iter_idx * ELEMS_PER_ITER;

#pragma unroll
            for (int j = 0; j < ELEMS_PER_ITER; j++) {
                float in_val = __half2float(input[offset + idx + j]);
                float weight_val = __half2float(weight[idx + j]);
                float result = in_val * inv_rms * (1.0f + weight_val);
                output[offset + idx + j] = __float2half(result);
            }
        }
    }
}

// ============================================================================
// Host function
// ============================================================================

torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight, float eps) {
    // Check inputs
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");

    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");

    const int hidden_size = input.size(-1);
    TORCH_CHECK(weight.size(0) == hidden_size, "weight size must match hidden_size");

    // Ensure contiguous
    input = input.contiguous();
    weight = weight.contiguous();

    // Create output
    auto output = torch::empty_like(input);

    // Flatten batch dimensions
    const int num_tokens = input.numel() / hidden_size;

    // Use specialized kernel for Gemma hidden_size (2048)
    if (hidden_size == 2048) {
        const int threads = 256;
        const int blocks = num_tokens;

        if (input.scalar_type() == at::ScalarType::Float) {
            rmsnorm_kernel_f32<2048><<<blocks, threads>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                num_tokens,
                eps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (input.scalar_type() == at::ScalarType::BFloat16) {
            rmsnorm_kernel_bf16<2048><<<blocks, threads>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                num_tokens,
                eps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (input.scalar_type() == at::ScalarType::Half) {
            rmsnorm_kernel_f16<2048><<<blocks, threads>>>(
                reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
                num_tokens,
                eps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            TORCH_CHECK(false, "Unsupported dtype for RMSNorm kernel");
        }
    } else {
        TORCH_CHECK(false, "Only hidden_size=2048 is supported in specialized kernel");
    }

    return output;
}
