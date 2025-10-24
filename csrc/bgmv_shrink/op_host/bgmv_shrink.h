#pragma once
#include <ATen/core/Tensor.h>

namespace vllm_ascend
{
    namespace npu_kernel
    {
        extern void bgmv_shrink(at::Tensor& x,
                                at::Tensor& weight,
                                at::Tensor& indices,
                                at::Tensor& y,
                                double      scale);
    }  // namespace npu_kernel
}  // namespace vllm_ascend