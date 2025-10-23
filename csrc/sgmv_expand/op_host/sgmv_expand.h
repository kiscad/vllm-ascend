#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>

namespace vllm_ascend {
namespace npu_kernel {
extern at::Tensor sgmv_expand(
    at::Tensor &x,
    at::Tensor &weight,
    at::Tensor &lora_indices,
    at::Tensor &seq_len,
    at::Tensor &y,
    int64_t slice_offset,
    int64_t slice_size
);
} // namespace npu_kernel

namespace meta {
at::Tensor sgmv_expand_meta(
    at::Tensor &x,
    at::Tensor &weight,
    at::Tensor &lora_indices,
    at::Tensor &seq_len,
    at::Tensor &y,
    int64_t slice_offset,
    int64_t slice_size);
} // namespace meta
} // namespace vllm_ascend