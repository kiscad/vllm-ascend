#include "bgmv_expand/op_host/bgmv_expand.h"
#include "bgmv_shrink/op_host/bgmv_shrink.h"
#include "get_masked_input_and_mask/op_host/get_masked_input_and_mask.h"
#include "mla_preprocess/op_host/mla_preprocess.h"
#include "rotary_embedding/op_host/rotary_embedding.h"
#include "sgmv_expand/op_host/sgmv_expand.h"
#include "sgmv_shrink/op_host/sgmv_shrink.h"
#include "swap_blocks/op_host/swap_blocks.h"
#include "weak_ref_tensor/op_host/weak_ref_tensor.h"
#include "utils/utils.h"

TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)
{
    // vLLM-Ascend custom ops
    ops.def(
        "bgmv_expand(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");
    ops.impl("bgmv_expand", c10::kPrivateUse1, &vllm_ascend::npu_kernel::bgmv_expand);

    ops.def("bgmv_shrink(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y, float scale) -> ()");
    ops.impl("bgmv_shrink", c10::kPrivateUse1, &vllm_ascend::npu_kernel::bgmv_shrink);

    ops.def(
        "get_masked_input_and_mask(Tensor input, "
        "                         int org_vocab_start_index, "
        "                         int org_vocab_end_index, "
        "                         int num_org_vocab_padding, "
        "                         int added_vocab_start_index, "
        "                         int added_vocab_end_index) -> (Tensor masked_input, Tensor mask)");
    ops.impl("get_masked_input_and_mask", c10::kPrivateUse1, &vllm_ascend::npu_kernel::get_masked_input_and_mask);

    ops.def(
        "mla_preprocess(Tensor hiddenState, Tensor wdqkv,"
        "               Tensor descale0, Tensor gamma1, Tensor beta1, Tensor wuq, Tensor descale1,"
        "               Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache,"
        "               Tensor kv_cache_rope, Tensor slotmapping, Tensor quant_scale0,"
        "               Tensor quant_offset0, Tensor bias0, Tensor quant_scale1, Tensor quant_offset1,"
        "               Tensor bias1, Tensor? ctkv_scale, Tensor? q_nope_scale, str? cache_mode,"
        "               str? quant_mode, Tensor! q_out0, Tensor! kv_cache_out0, Tensor! q_out1,"
        "               Tensor! kv_cache_out1) -> (Tensor q_out0, Tensor kv_cache_out0,"
        "                                          Tensor q_out1, Tensor kv_cache_out1)"
    );
    ops.impl("mla_preprocess", c10::kPrivateUse1, &vllm_ascend::npu_kernel::mla_preprocess);

    // Rotary embedding
    // Apply GPT-NeoX style rotary embedding to query and key.
    ops.def(
        "rotary_embedding(Tensor positions, Tensor! query,"
        "                 Tensor! key, int head_size,"
        "                 Tensor cos_sin_cache, bool is_neox) -> (Tensor query, Tensor key)");
    ops.impl("rotary_embedding", c10::kPrivateUse1, &vllm_ascend::npu_kernel::rotary_embedding);

    ops.def(
        "sgmv_expand(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");
    ops.impl("sgmv_expand", c10::kPrivateUse1, &vllm_ascend::npu_kernel::sgmv_expand);

    ops.def("sgmv_shrink(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y, float scale) -> ()");
    ops.impl("sgmv_shrink", c10::kPrivateUse1, &vllm_ascend::npu_kernel::sgmv_shrink);

    ops.def("swap_blocks(Tensor! x, Tensor! y, Tensor z) -> ()");
    ops.impl("swap_blocks", c10::kPrivateUse1, &vllm_ascend::npu_kernel::swap_blocks);

    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
    ops.impl("weak_ref_tensor", c10::kPrivateUse1, &vllm_ascend::npu_kernel::weak_ref_tensor);
}


/*
 * How to write a meta implementation for a custom operator (meta kernel):
 *
 * Meta implementations are used for shape and dtype inference, tracing, and export.
 * They do NOT perform any real computation or allocate device memory.
 * Instead, they return empty tensors with the correct shapes, dtypes, and device types.
 *
 * Steps to write a meta implementation:
 * 1. The function signature should match the operator's schema, but only use the arguments
 *    necessary to infer output shapes and dtypes.
 * 2. Use input tensor shapes, dtypes, and any relevant arguments to compute the output shapes.
 * 3. Return empty tensors (e.g., at::empty_symint, at::empty_like) with the correct shape and dtype.
 * 4. Do NOT perform any real computation or data movement.
 * 5. Register the meta implementation with the "Meta" dispatch key using TORCH_LIBRARY_IMPL or similar.
 *
 * Example:
 *   std::tuple<at::Tensor, at::Tensor> my_op_meta(
 *       at::Tensor &input, int64_t some_param) {
 *     // Infer output shape based on input and parameters
 *     auto out_shape = ...;
 *     at::Tensor out = at::empty_symint(out_shape, input.options());
 *     // Return empty tensor(s) with correct shape/dtype
 *     return {out, ...};
 *   }
 *
 * See below for real examples.
 */
namespace {
// Register the meta implementations of the custom kernels for symbolic tracing; this also
// ensures the custom kernels can be captured into aclgraph executions.
TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _ascend), Meta, ops) {
    // Rotary embedding meta implementation
    ops.impl("rotary_embedding", &vllm_ascend::meta::rotary_embedding_meta);
    // Masked input and mask meta implementation
    ops.impl("get_masked_input_and_mask", &vllm_ascend::meta::get_masked_input_and_mask_meta);
    // Bgmv expand
    ops.impl("bgmv_expand", &vllm_ascend::meta::bgmv_expand_meta);
    // Sgmv expand
    ops.impl("sgmv_expand", &vllm_ascend::meta::sgmv_expand_meta);
    // MLA preprocess
    ops.impl("mla_preprocess", &vllm_ascend::meta::mla_preprocess_meta);
}
} // namespace
