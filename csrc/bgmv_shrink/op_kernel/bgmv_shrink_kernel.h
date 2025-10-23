#pragma once

#include "utils/types.h"

namespace vllm_ascend {
namespace npu_kernel {

    extern void bgmv_shrink_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *indices,
        uint32_t indicesSize,
        void *y, 
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t input_hidden_dim,
        uint32_t lora_rank,
        float scale);

} // namespace npu_kernel
} // namespace vllm_ascend
