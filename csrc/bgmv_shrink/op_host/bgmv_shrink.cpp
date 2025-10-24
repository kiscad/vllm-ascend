#include "../op_kernel/bgmv_shrink_kernel.h"
#include "../../utils/host_common.h"

namespace vllm_ascend
{
    namespace npu_kernel
    {

        void bgmv_shrink(
                at::Tensor& x, at::Tensor& weight, at::Tensor& indices, at::Tensor& y, double scale)
        {
            at::ScalarType scalar_type = x.scalar_type();
            TORCH_CHECK(scalar_type == c10::kHalf || scalar_type == c10::kBFloat16,
                        "only support half and bf16");
            TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
            TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                        "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, "
                        "hidden_out, hidden_in]");
            TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
            TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
            TORCH_CHECK(x.size(0) == y.size(0) && x.size(0) == indices.size(0),
                        "the first dimension of x, y, indices should be same");
            TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
            void*                     x_ptr              = x.data_ptr();
            void*                     weight_ptr         = weight.data_ptr();
            void*                     indices_ptr        = indices.data_ptr();
            int                       indices_size       = indices.size(0);
            void*                     y_ptr              = y.data_ptr();
            int                       batch_size         = x.size(0);
            int                       input_hidden_token = x.size(1);
            uint32_t                  lora_rank          = y.size(1);
            float                     scale_f            = static_cast<float>(scale);
            aclrtStream               stream             = c10_npu::getCurrentNPUStream().stream();
            at_npu::native::OpCommand cmd;
            cmd.Name("bgmv_shrink");
            cmd.SetCustomHandler(
                    [scalar_type,
                     stream,
                     x_ptr,
                     weight_ptr,
                     indices_ptr,
                     indices_size,
                     y_ptr,
                     batch_size,
                     input_hidden_token,
                     lora_rank,
                     scale_f]() -> int
                    {
                        auto    dtype     = get_dtype_from_torch(scalar_type);
                        int     device_id = 0;
                        int64_t aiv_num   = 0;
                        TORCH_CHECK(aclGetDeviceCapability(device_id,
                                                           ACL_DEVICE_INFO_VECTOR_CORE_NUM,
                                                           &aiv_num) == ACL_SUCCESS);
                        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
                        TORCH_CHECK("num_tokens_per_core != 0",
                                    "num_tokens_per_core should not be 0");
                        bgmv_shrink_impl(dtype,
                                         stream,
                                         x_ptr,
                                         weight_ptr,
                                         indices_ptr,
                                         indices_size,
                                         y_ptr,
                                         batch_size,
                                         num_tokens_per_core,
                                         input_hidden_token,
                                         lora_rank,
                                         scale_f);
                        return 0;
                    });
            cmd.Run();
            return;
        }

    }  // namespace npu_kernel
}  // namespace vllm_ascend
