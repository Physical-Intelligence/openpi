#include <c10/cuda/CUDAException.h>  // For error checking
#include <cuda.h>
#include <cuda_runtime.h>

#include "ops.h"  // Include header to ensure signatures match

template <typename scalar_t>
__global__ void add_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
                           const scalar_t value, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + value;
    }
}

// This matches the signature in ops.h
torch::Tensor add_cuda(torch::Tensor input, float value) {
    // 1. Check Device
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

    // 2. Check Layout (Critical for kernels using linear index)
    input = input.contiguous();

    auto output = torch::zeros_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "add_cuda", ([&] {
                                   add_kernel<scalar_t><<<blocks, threads>>>(
                                       input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                                       static_cast<scalar_t>(value), size);
                                   C10_CUDA_KERNEL_LAUNCH_CHECK();
                               }));

    return output;
}
