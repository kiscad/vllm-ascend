#pragma once

namespace vllm_ascend {
namespace npu_kernel {
extern std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess(
    const at::Tensor &hiddenState,
    const at::Tensor &wdqkv,
    const at::Tensor &descale0,
    const at::Tensor &gamma1,
    const at::Tensor &beta1,
    const at::Tensor &wuq,
    const at::Tensor &descale1,
    const at::Tensor &gamma2,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const at::Tensor &wuk,
    const at::Tensor &kv_cache,
    const at::Tensor &kv_cache_rope,
    const at::Tensor &slotmapping,
    const at::Tensor &quant_scale0,
    const at::Tensor &quant_offset0,
    const at::Tensor &bias0,
    const at::Tensor &quant_scale1,
    const at::Tensor &quant_offset1,
    const at::Tensor &bias1,
    const c10::optional<at::Tensor> &ctkv_scale,
    const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode,
    c10::optional<c10::string_view> quant_mode,
    at::Tensor &q_out0,
    at::Tensor &kv_cache_out0,
    at::Tensor &q_out1,
    at::Tensor &kv_cache_out1
);
} // namespace npu_kernel

namespace meta {
extern std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess_meta(
    const at::Tensor &hiddenState,
    const at::Tensor &wdqkv,
    const at::Tensor &descale0,
    const at::Tensor &gamma1,
    const at::Tensor &beta1,
    const at::Tensor &wuq,
    const at::Tensor &descale1,
    const at::Tensor &gamma2,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const at::Tensor &wuk,
    const at::Tensor &kv_cache,
    const at::Tensor &kv_cache_rope,
    const at::Tensor &slotmapping,
    const at::Tensor &quant_scale0,
    const at::Tensor &quant_offset0,
    const at::Tensor &bias0,
    const at::Tensor &quant_scale1,
    const at::Tensor &quant_offset1,
    const at::Tensor &bias1,
    const c10::optional<at::Tensor> &ctkv_scale,
    const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode,
    c10::optional<c10::string_view> quant_mode,
    at::Tensor &q_out0,
    at::Tensor &kv_cache_out0,
    at::Tensor &q_out1,
    at::Tensor &kv_cache_out1);
} // namespace meta
} // namespace vllm_ascend