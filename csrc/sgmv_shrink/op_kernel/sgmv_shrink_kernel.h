#pragma once

#include "../../utils/types.h"

namespace vllm_ascend
{
    namespace npu_kernel
    {

        extern void sgmv_shrink_impl(AscendType type,
                                     void*      stream,
                                     void*      x,
                                     void*      weight,
                                     void*      loraIndices,
                                     uint32_t   loraIndicesSize,
                                     void*      seqLen,
                                     uint32_t   seqLenSize,
                                     void*      y,
                                     uint32_t   batch_size,
                                     uint32_t   num_tokens_per_core,
                                     uint32_t   input_hidden_dim,
                                     uint32_t   lora_rank,
                                     float      scale);

    }  // namespace npu_kernel
}  // namespace vllm_ascend
