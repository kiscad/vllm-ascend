#include "../op_kernel/sgmv_expand_kernel.h"
#include "../../utils/host_common.h"

namespace vllm_ascend {
namespace npu_kernel {
at::Tensor sgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                       at::Tensor &y, int64_t slice_offset, int64_t slice_size)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");
    TORCH_CHECK(slice_offset >= 0, "slice offset should be no smaller than 0");
    TORCH_CHECK((slice_size + slice_offset) <= y.size(1),
                "slice_size + slice_offset should be smaller than the second dimension of y")

    at::Tensor y_out = y;
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* lora_indices_ptr = lora_indices.data_ptr();
    void* seq_len_ptr = seq_len.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    int seq_len_size = seq_len.size(0);
    void* y_ptr = y.data_ptr();
    void* y_out_ptr = y_out.data_ptr();
    int batch_size = x.size(0);
    int lora_rank = x.size(1);
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                          batch_size, lora_rank, slice_offset, slice_size, output_full_dim]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        sgmv_expand_impl(dtype, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                         batch_size, num_tokens_per_core, lora_rank, slice_size, slice_offset, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}
} // namespace npu_kernel

namespace meta {
at::Tensor sgmv_expand_meta(
    at::Tensor &x,
    at::Tensor &weight,
    at::Tensor &lora_indices,
    at::Tensor &seq_len,
    at::Tensor &y,
    int64_t slice_offset,
    int64_t slice_size
) {
    at::Tensor y_out = at::empty_like(y);
    return y_out;
}
} // namespace meta
} // namespace vllm_ascend
