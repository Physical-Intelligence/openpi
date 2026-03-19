#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops.h"

// GELU activation with tanh approximation
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu_tanh(scalar_t x) {
    const scalar_t sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
    const scalar_t coeff = 0.044715f;

    scalar_t x_cubed = x * x * x;
    scalar_t inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    scalar_t tanh_inner = tanhf(inner);

    return 0.5f * x * (1.0f + tanh_inner);
}

// Specialization for half precision
__device__ __forceinline__ __half gelu_tanh(__half x) {
    float x_float = __half2float(x);
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    float x_cubed = x_float * x_float * x_float;
    float inner = sqrt_2_over_pi * (x_float + coeff * x_cubed);
    float tanh_inner = tanhf(inner);

    return __float2half(0.5f * x_float * (1.0f + tanh_inner));
}

// Specialization for bfloat16
__device__ __forceinline__ __nv_bfloat16 gelu_tanh(__nv_bfloat16 x) {
    float x_float = __bfloat162float(x);
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    float x_cubed = x_float * x_float * x_float;
    float inner = sqrt_2_over_pi * (x_float + coeff * x_cubed);
    float tanh_inner = tanhf(inner);

    return __float2bfloat16(0.5f * x_float * (1.0f + tanh_inner));
}

// Fused Bias + GELU kernel
// Combines: output = GELU(input + bias)
// Input shape: [batch_size, seq_len, features] (or any shape that ends with features)
// Bias shape: [features]
template <typename scalar_t>
__global__ void fused_bias_gelu_kernel(const scalar_t* __restrict__ input,
                                       const scalar_t* __restrict__ bias,
                                       scalar_t* __restrict__ output, const int total_elements,
                                       const int features) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // Get the feature index for this element
        const int feature_idx = idx % features;

        // Add bias and apply GELU
        scalar_t val = input[idx] + bias[feature_idx];
        output[idx] = gelu_tanh(val);
    }
}

// Optimized kernel using vectorized loads (float4 for better memory bandwidth)
template <typename scalar_t>
__global__ void fused_bias_gelu_kernel_vectorized(const scalar_t* __restrict__ input,
                                                  const scalar_t* __restrict__ bias,
                                                  scalar_t* __restrict__ output,
                                                  const int num_tokens, const int features) {
    // Each thread processes 4 elements
    const int token_idx = blockIdx.x;
    const int feature_base = threadIdx.x * 4;

    if (token_idx < num_tokens && feature_base + 3 < features) {
        const int base_idx = token_idx * features + feature_base;

        // Vectorized load of input (4 elements at once)
        const scalar_t* input_ptr = input + base_idx;
        scalar_t in_vals[4];

#pragma unroll
        for (int i = 0; i < 4; i++) {
            in_vals[i] = input_ptr[i];
        }

        // Load bias values
        const scalar_t* bias_ptr = bias + feature_base;
        scalar_t bias_vals[4];

#pragma unroll
        for (int i = 0; i < 4; i++) {
            bias_vals[i] = bias_ptr[i];
        }

        // Compute bias + GELU
        scalar_t out_vals[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            scalar_t val = in_vals[i] + bias_vals[i];
            out_vals[i] = gelu_tanh(val);
        }

        // Vectorized store
        scalar_t* output_ptr = output + base_idx;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            output_ptr[i] = out_vals[i];
        }
    }

    // Handle remaining elements (when features % 4 != 0)
    if (token_idx < num_tokens) {
        const int features_rounded = (features / 4) * 4;
        for (int feature_idx = features_rounded + threadIdx.x; feature_idx < features;
             feature_idx += blockDim.x) {
            const int idx = token_idx * features + feature_idx;
            scalar_t val = input[idx] + bias[feature_idx];
            output[idx] = gelu_tanh(val);
        }
    }
}

// ============================================================================
// SPECIALIZED KERNEL FOR SIGLIP: FEATURES = 4304
// Optimizations:
// - Compile-time known dimensions (template specialization)
// - float4 vectorized loads/stores for float32 (4x bandwidth)
// - Full loop unrolling
// - Optimal thread configuration (256 threads per block)
// ============================================================================

template <int FEATURES>
__global__ void fused_bias_gelu_kernel_siglip_f32(const float* __restrict__ input,
                                                  const float* __restrict__ bias,
                                                  float* __restrict__ output,
                                                  const int num_tokens) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid = threadIdx.x;  // 0-255
    const int input_offset = token_idx * FEATURES;

    // Use float4 for vectorized access (4 floats = 16 bytes)
    // FEATURES = 4304, so we have 4304/4 = 1076 float4 vectors
    // With 256 threads, each thread processes 1076/256 ≈ 4.2 vectors
    constexpr int NUM_VEC4 = FEATURES / 4;                   // 1076
    constexpr int VEC4_PER_THREAD = (NUM_VEC4 + 255) / 256;  // 5 (rounded up)

// Each thread processes VEC4_PER_THREAD float4 vectors
#pragma unroll
    for (int i = 0; i < VEC4_PER_THREAD; i++) {
        const int vec_idx = tid + i * 256;

        if (vec_idx < NUM_VEC4) {
            const int feature_idx = vec_idx * 4;

            // Vectorized load (16-byte aligned)
            float4 in_vec = *reinterpret_cast<const float4*>(&input[input_offset + feature_idx]);
            float4 bias_vec = *reinterpret_cast<const float4*>(&bias[feature_idx]);

            // Compute bias + GELU for all 4 elements
            float4 out_vec;
            out_vec.x = gelu_tanh(in_vec.x + bias_vec.x);
            out_vec.y = gelu_tanh(in_vec.y + bias_vec.y);
            out_vec.z = gelu_tanh(in_vec.z + bias_vec.z);
            out_vec.w = gelu_tanh(in_vec.w + bias_vec.w);

            // Vectorized store (16-byte aligned)
            *reinterpret_cast<float4*>(&output[input_offset + feature_idx]) = out_vec;
        }
    }
}

// Specialized kernel for bfloat16 - use float32 for computation
// This is faster because bf16 intrinsics are less optimized
template <int FEATURES>
__global__ void fused_bias_gelu_kernel_siglip_bf16(const __nv_bfloat16* __restrict__ input,
                                                   const __nv_bfloat16* __restrict__ bias,
                                                   __nv_bfloat16* __restrict__ output,
                                                   const int num_tokens) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid = threadIdx.x;
    const int input_offset = token_idx * FEATURES;

    // Process 4 bf16 elements at a time, convert to float for computation
    constexpr int ELEMS_PER_ITER = 4;
    constexpr int NUM_ITERS = FEATURES / ELEMS_PER_ITER;       // 4304/4 = 1076
    constexpr int ITERS_PER_THREAD = (NUM_ITERS + 255) / 256;  // 5

#pragma unroll
    for (int i = 0; i < ITERS_PER_THREAD; i++) {
        const int iter_idx = tid + i * 256;

        if (iter_idx < NUM_ITERS) {
            const int feature_idx = iter_idx * ELEMS_PER_ITER;

            // Load 4 bf16 values and convert to float
            const __nv_bfloat16* in_ptr = &input[input_offset + feature_idx];
            const __nv_bfloat16* bias_ptr = &bias[feature_idx];
            __nv_bfloat16* out_ptr = &output[input_offset + feature_idx];

#pragma unroll
            for (int j = 0; j < ELEMS_PER_ITER; j++) {
                // Convert to float for computation (faster than bf16 arithmetic)
                float val = __bfloat162float(in_ptr[j]) + __bfloat162float(bias_ptr[j]);
                float result = gelu_tanh(val);
                out_ptr[j] = __float2bfloat16(result);
            }
        }
    }
}

// Specialized kernel for float16
template <int FEATURES>
__global__ void fused_bias_gelu_kernel_siglip_f16(const __half* __restrict__ input,
                                                  const __half* __restrict__ bias,
                                                  __half* __restrict__ output,
                                                  const int num_tokens) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid = threadIdx.x;
    const int input_offset = token_idx * FEATURES;

    // Process 8 fp16 elements at a time (16 bytes)
    constexpr int ELEMS_PER_LOAD = 8;
    constexpr int NUM_LOADS = FEATURES / ELEMS_PER_LOAD;       // 4304/8 = 538
    constexpr int LOADS_PER_THREAD = (NUM_LOADS + 255) / 256;  // 3

#pragma unroll
    for (int i = 0; i < LOADS_PER_THREAD; i++) {
        const int load_idx = tid + i * 256;

        if (load_idx < NUM_LOADS) {
            const int feature_idx = load_idx * ELEMS_PER_LOAD;

            const __half* in_ptr = &input[input_offset + feature_idx];
            const __half* bias_ptr = &bias[feature_idx];
            __half* out_ptr = &output[input_offset + feature_idx];

#pragma unroll
            for (int j = 0; j < ELEMS_PER_LOAD; j++) {
                __half val = __hadd(in_ptr[j], bias_ptr[j]);
                out_ptr[j] = gelu_tanh(val);
            }
        }
    }
}

// Host function
torch::Tensor fused_bias_gelu_cuda(torch::Tensor input, torch::Tensor bias) {
    // 1. Check Device
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA");

    // 2. Check dimensions
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

    const int features = bias.size(0);
    TORCH_CHECK(input.size(-1) == features, "Input last dimension must match bias size");

    // 3. Ensure contiguous layout
    input = input.contiguous();
    bias = bias.contiguous();

    // 4. Create output tensor
    auto output = torch::empty_like(input);

    const int total_elements = input.numel();
    const int num_tokens = total_elements / features;

    // Get current CUDA stream for torch.compile compatibility
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 5. Choose kernel based on feature size
    const int max_threads_per_block = 1024;

    // Check if we should use the specialized SiGLIP kernel (features = 4304)
    if (features == 4304) {
        // Use optimized SiGLIP-specialized kernel
        const int threads = 256;
        const int blocks = num_tokens;

        if (input.scalar_type() == at::ScalarType::Float) {
            fused_bias_gelu_kernel_siglip_f32<4304>
                <<<blocks, threads, 0, stream>>>(input.data_ptr<float>(), bias.data_ptr<float>(),
                                      output.data_ptr<float>(), num_tokens);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (input.scalar_type() == at::ScalarType::BFloat16) {
            fused_bias_gelu_kernel_siglip_bf16<4304><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), num_tokens);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (input.scalar_type() == at::ScalarType::Half) {
            fused_bias_gelu_kernel_siglip_f16<4304><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(output.data_ptr<at::Half>()), num_tokens);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            TORCH_CHECK(false, "Unsupported dtype for specialized SiGLIP kernel");
        }
    } else if (features % 4 == 0 && features >= 256 && (features / 4) <= max_threads_per_block) {
        // Use generic vectorized kernel for other large feature dimensions
        const int threads = features / 4;
        const int blocks = num_tokens;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "fused_bias_gelu_cuda_vectorized", ([&] {
                fused_bias_gelu_kernel_vectorized<scalar_t>
                    <<<blocks, threads, 0, stream>>>(input.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                                          output.data_ptr<scalar_t>(), num_tokens, features);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }));
    } else {
        // Use simple kernel for all other cases
        const int threads = 256;
        const int blocks = (total_elements + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
            "fused_bias_gelu_cuda", ([&] {
                fused_bias_gelu_kernel<scalar_t>
                    <<<blocks, threads, 0, stream>>>(input.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                                          output.data_ptr<scalar_t>(), total_elements, features);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }));
    }

    return output;
}
