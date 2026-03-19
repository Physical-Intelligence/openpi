#pragma once
#include <torch/extension.h>

#include <tuple>

torch::Tensor add_cuda(torch::Tensor input, float value);

torch::Tensor fused_bias_gelu_cuda(torch::Tensor input, torch::Tensor bias);

torch::Tensor fused_geglu_cuda(torch::Tensor gate, torch::Tensor up);

torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight, float eps);

std::tuple<torch::Tensor, torch::Tensor> fused_add_layernorm_cuda(torch::Tensor x,
                                                                  torch::Tensor residual,
                                                                  torch::Tensor gamma,
                                                                  torch::Tensor beta, float eps);