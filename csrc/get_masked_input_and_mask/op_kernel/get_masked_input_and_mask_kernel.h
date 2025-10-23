#pragma once

#include "../../utils/types.h"

namespace vllm_ascend {
namespace npu_kernel {

    extern void get_masked_input_and_mask_impl(
        void* stream,
        void* input,
        void* masked_input,
        void* mask_out,
        const int64_t org_vocab_start_index,
        const int64_t org_vocab_end_index,
        const int64_t num_org_vocab_padding, 
        const int64_t added_vocab_start_index,
        const int64_t added_vocab_end_index,
        const int64_t size,
        const uint32_t loop_cnt,
        const uint32_t aiv_num
    );

} // namespace npu_kernel
} // namespace vllm_ascend
