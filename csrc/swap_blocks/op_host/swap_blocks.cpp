#include "../../utils/host_common.h"

namespace vllm_ascend
{
    namespace op_swap_block
    {
        void swap_blocks_impl(at::Tensor&       src,
                              at::Tensor&       dst,
                              const at::Tensor& block_mapping,
                              aclrtStream       stream)
        {
            torch::Device   src_device = src.device();
            torch::Device   dst_device = dst.device();
            aclrtMemcpyKind memcpy_type;

            if ((!src_device.is_cpu()) && (!dst_device.is_cpu()))
            {
                TORCH_CHECK(src_device.index() == dst_device.index(),
                            "src and dst must be on the same npu");
                memcpy_type = ACL_MEMCPY_DEVICE_TO_DEVICE;
            }
            else if ((!src_device.is_cpu()) && dst_device.is_cpu())
            {
                memcpy_type = ACL_MEMCPY_DEVICE_TO_HOST;
            }
            else if (src_device.is_cpu() && (!dst_device.is_cpu()))
            {
                memcpy_type = ACL_MEMCPY_HOST_TO_DEVICE;
            }
            else
            {
                TORCH_CHECK(false,
                            "Invalid device combination, src tensor device: ",
                            src_device,
                            ", dst tensor device: ",
                            dst_device);
            }

            TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

            char* src_ptr = static_cast<char*>(src.data_ptr());
            char* dst_ptr = static_cast<char*>(dst.data_ptr());

            const int64_t block_size_in_bytes = src.element_size() * src.stride(0);

            const int64_t num_blocks    = block_mapping.size(0);
            const int64_t max_src_block = src.size(0);
            const int64_t max_dst_block = dst.size(0);
            for (size_t i = 0; i < num_blocks; i++)
            {
                int64_t src_block_number = block_mapping[i][0].item<int64_t>();
                int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
                TORCH_CHECK(src_block_number >= 0 && src_block_number <= max_src_block,
                            "src block index ",
                            src_block_number,
                            " out of range (max: ",
                            max_src_block,
                            ")");
                TORCH_CHECK(dst_block_number >= 0 && dst_block_number <= max_dst_block,
                            "dst block index ",
                            dst_block_number,
                            " out of range (max: ",
                            max_dst_block,
                            ")");

                int64_t src_offset = src_block_number * block_size_in_bytes;
                int64_t dst_offset = dst_block_number * block_size_in_bytes;

                aclrtMemcpyAsync(dst_ptr + dst_offset,
                                 block_size_in_bytes,
                                 src_ptr + src_offset,
                                 block_size_in_bytes,
                                 memcpy_type,
                                 stream);
            }
        }
    }  // namespace op_swap_block

    namespace npu_kernel
    {

        void swap_blocks(at::Tensor& x, at::Tensor& y, const at::Tensor& z)
        {
            const c10_npu::OptionalNPUGuard npuGuard((!x.device().is_cpu()) ? x.device()
                                                                            : y.device());
            aclrtStream                     stream = c10_npu::getCurrentNPUStream().stream();
            op_swap_block::swap_blocks_impl(x, y, z, stream);
            return;
        }

    }  // namespace npu_kernel
}  // namespace vllm_ascend
