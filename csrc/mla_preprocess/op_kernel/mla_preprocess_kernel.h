#pragma once

namespace vllm_ascend {
namespace npu_kernel {
    extern void mla_preprocess_impl(
        void* stream,
        void* hidden_state,
        void* quant_scale1,
        void* quant_offset1,
        void* wdqkv,
        void* bias1,
        void* gamma2,
        void* beta2,
        void* quant_scale2,
        void* quant_offset2,
        void* gamma3,
        void* sin1,
        void* cos1,
        void* sin2,
        void* cos2,
        void* keycache,
        void* slot_mapping,
        void* wuq,
        void* bias2,
        void* wuk,
        void* descale1,
        void* descale2,
        void* ctkv_scale,
        void* qnope_scale,
        void* q,
        void* keycache_out,
        void* q2,
        void* keycache_out2,
        void* workspace,
        void* tiling,
        const uint32_t block_dim
    );
} // namespace npu_kernel
} // namespace vllm_ascend
