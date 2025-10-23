#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>

#include "bgmv_expand/op_host/bgmv_expand.h"
#include "get_masked_input_and_mask/op_host/get_masked_input_and_mask.h"
#include "mla_preprocess/op_host/mla_preprocess.h"
#include "rotary_embedding/op_host/rotary_embedding.h"
#include "sgmv_expand/op_host/sgmv_expand.h"
#include "utils.h"

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
  // Register the meta implementations of the custom kernels for symbolic tracing, this will also
  // the custom kernel been captured into aclgraph
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
}
