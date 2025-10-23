#pragma once
#include <ATen/core/Tensor.h>

namespace vllm_ascend {
namespace npu_kernel {
extern void swap_blocks(at::Tensor &x, at::Tensor &y, const at::Tensor &z);
} // namespace npu_kernel
} // namespace vllm_ascend