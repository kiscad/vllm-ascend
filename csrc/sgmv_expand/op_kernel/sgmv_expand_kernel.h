#pragma once

#include "../../utils/types.h"

namespace vllm_ascend
{
    namespace npu_kernel
    {

        extern void sgmv_expand_impl(AscendType type,
                                     void*      stream,
                                     void*      x,
                                     void*      weight,
                                     void*      loraIndices,
                                     uint32_t   loraIndicesSize,
                                     void*      seqLen,
                                     uint32_t   seqLenSize,
                                     void*      y,
                                     void*      y_out,
                                     uint32_t   batch_size,
                                     uint32_t   num_tokens_per_core,
                                     uint32_t   lora_rank,
                                     uint32_t   output_hidden_dim,
                                     uint32_t   slice_offset,
                                     uint32_t   output_full_dim);

    }  // namespace npu_kernel
}  // namespace vllm_ascend
