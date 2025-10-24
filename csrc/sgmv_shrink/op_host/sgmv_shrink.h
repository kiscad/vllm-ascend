#pragma once
#include <ATen/core/Tensor.h>

namespace vllm_ascend
{
    namespace npu_kernel
    {

        extern void sgmv_shrink(at::Tensor& x,
                                at::Tensor& weight,
                                at::Tensor& lora_indices,
                                at::Tensor& seq_len,
                                at::Tensor& y,
                                double      scale);

    }  // namespace npu_kernel
}  // namespace vllm_ascend