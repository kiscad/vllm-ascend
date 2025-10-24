#pragma once
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>

namespace vllm_ascend
{
    namespace npu_kernel
    {
        extern std::tuple<at::Tensor, at::Tensor> rotary_embedding(at::Tensor& positions,
                                                                   at::Tensor& query,
                                                                   at::Tensor& key,
                                                                   int64_t     head_size,
                                                                   at::Tensor& cos_sin_cache,
                                                                   bool        is_neox);
    }  // namespace npu_kernel

    namespace meta
    {
        std::tuple<at::Tensor, at::Tensor> rotary_embedding_meta(at::Tensor& positions,
                                                                 at::Tensor& query,
                                                                 at::Tensor& key,
                                                                 int64_t     head_size,
                                                                 at::Tensor& cos_sin_cache,
                                                                 bool        is_neox);
    }  // namespace meta
}  // namespace vllm_ascend