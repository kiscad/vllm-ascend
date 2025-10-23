#pragma once
#include <ATen/core/Tensor.h>

namespace vllm_ascend {
namespace npu_kernel {

at::Tensor weak_ref_tensor(at::Tensor& tensor);

} // namespace npu_kernel
} // namespace vllm_ascend