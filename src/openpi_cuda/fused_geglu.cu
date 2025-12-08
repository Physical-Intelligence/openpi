#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops.h"

// ============================================================================
// FUSED GEGLU KERNEL (GELU Gated Linear Unit)
// Combines: output = GELU(gate) * up
// This saves one memory round-trip by avoiding writing the GELU result
// ============================================================================

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

// ============================================================================
// OPTIMIZED FUSED GEGLU KERNEL (Float32 with vectorized loads)
// ============================================================================

__global__ void fused_geglu_kernel_f32_vec4(const float* __restrict__ gate,
                                            const float* __restrict__ up,
                                            float* __restrict__ output, const int total_elements) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < total_elements) {
        // Vectorized load (16 bytes each)
        float4 gate_vec = *reinterpret_cast<const float4*>(&gate[idx]);
        float4 up_vec = *reinterpret_cast<const float4*>(&up[idx]);

        // Apply GELU to gate and multiply with up
        float4 out_vec;
        out_vec.x = gelu_tanh(gate_vec.x) * up_vec.x;
        out_vec.y = gelu_tanh(gate_vec.y) * up_vec.y;
        out_vec.z = gelu_tanh(gate_vec.z) * up_vec.z;
        out_vec.w = gelu_tanh(gate_vec.w) * up_vec.w;

        // Vectorized store (16 bytes)
        *reinterpret_cast<float4*>(&output[idx]) = out_vec;
    } else if (idx < total_elements) {
        // Handle remaining elements
        for (int i = idx; i < total_elements && i < idx + 4; i++) {
            output[i] = gelu_tanh(gate[i]) * up[i];
        }
    }
}

// ============================================================================
// OPTIMIZED FUSED GEGLU KERNEL (BFloat16)
// Converts to float for computation (faster than native bf16 arithmetic)
// ============================================================================

__global__ void fused_geglu_kernel_bf16(const __nv_bfloat16* __restrict__ gate,
                                        const __nv_bfloat16* __restrict__ up,
                                        __nv_bfloat16* __restrict__ output,
                                        const int total_elements) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < total_elements) {
        // Load 4 bf16 values
        const __nv_bfloat16* gate_ptr = &gate[idx];
        const __nv_bfloat16* up_ptr = &up[idx];
        __nv_bfloat16* out_ptr = &output[idx];

#pragma unroll
        for (int i = 0; i < 4; i++) {
            // Convert to float for computation (faster than bf16 intrinsics)
            float gate_val = __bfloat162float(gate_ptr[i]);
            float up_val = __bfloat162float(up_ptr[i]);
            float result = gelu_tanh(gate_val) * up_val;
            out_ptr[i] = __float2bfloat16(result);
        }
    } else if (idx < total_elements) {
        // Handle remaining elements
        for (int i = idx; i < total_elements && i < idx + 4; i++) {
            float gate_val = __bfloat162float(gate[i]);
            float up_val = __bfloat162float(up[i]);
            output[i] = __float2bfloat16(gelu_tanh(gate_val) * up_val);
        }
    }
}

// ============================================================================
// OPTIMIZED FUSED GEGLU KERNEL (Float16)
// ============================================================================

__global__ void fused_geglu_kernel_f16(const __half* __restrict__ gate,
                                       const __half* __restrict__ up, __half* __restrict__ output,
                                       const int total_elements) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    if (idx + 7 < total_elements) {
        // Load 8 fp16 values (16 bytes)
        const __half* gate_ptr = &gate[idx];
        const __half* up_ptr = &up[idx];
        __half* out_ptr = &output[idx];

#pragma unroll
        for (int i = 0; i < 8; i++) {
            out_ptr[i] = __hmul(gelu_tanh(gate_ptr[i]), up_ptr[i]);
        }
    } else if (idx < total_elements) {
        // Handle remaining elements
        for (int i = idx; i < total_elements && i < idx + 8; i++) {
            output[i] = __hmul(gelu_tanh(gate[i]), up[i]);
        }
    }
}

// ============================================================================
// Host function
// ============================================================================

torch::Tensor fused_geglu_cuda(torch::Tensor gate, torch::Tensor up) {
    // 1. Check Device
    TORCH_CHECK(gate.is_cuda(), "Gate tensor must be on CUDA");
    TORCH_CHECK(up.is_cuda(), "Up tensor must be on CUDA");

    // 2. Check dimensions
    TORCH_CHECK(gate.sizes() == up.sizes(), "Gate and up tensors must have the same shape");

    // 3. Ensure contiguous layout
    gate = gate.contiguous();
    up = up.contiguous();

    // 4. Create output tensor
    auto output = torch::empty_like(gate);

    const int total_elements = gate.numel();

    // 5. Launch kernel based on dtype
    const int threads = 256;

    if (gate.scalar_type() == at::ScalarType::Float) {
        // Process 4 floats per thread
        const int blocks = (total_elements + threads * 4 - 1) / (threads * 4);
        fused_geglu_kernel_f32_vec4<<<blocks, threads>>>(
            gate.data_ptr<float>(), up.data_ptr<float>(), output.data_ptr<float>(), total_elements);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (gate.scalar_type() == at::ScalarType::BFloat16) {
        // Process 4 bf16 per thread
        const int blocks = (total_elements + threads * 4 - 1) / (threads * 4);
        fused_geglu_kernel_bf16<<<blocks, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(up.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), total_elements);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (gate.scalar_type() == at::ScalarType::Half) {
        // Process 8 fp16 per thread
        const int blocks = (total_elements + threads * 8 - 1) / (threads * 8);
        fused_geglu_kernel_f16<<<blocks, threads>>>(
            reinterpret_cast<const __half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(up.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), total_elements);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        TORCH_CHECK(false,
                    "Unsupported dtype for GeGLU kernel. Supported: float32, float16, bfloat16");
    }

    return output;
}
