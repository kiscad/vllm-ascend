#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>

namespace vllm_ascend {
namespace npu_kernel {
extern at::Tensor bgmv_expand(
    at::Tensor &x,
    at::Tensor &weight,
    at::Tensor &indices,
    at::Tensor &y,
    int64_t slice_offset,
    int64_t slice_size
);
} // namespace npu_kernel

namespace meta {
extern at::Tensor bgmv_expand_meta(
    at::Tensor &x,
    at::Tensor &weight,
    at::Tensor &indices,
    at::Tensor &y,
    int64_t slice_offset,
    int64_t slice_size
);
} // namespace meta
} // namespace vllm_ascend