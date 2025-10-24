#include "../op_kernel/get_masked_input_and_mask_kernel.h"
#include "../../utils/host_common.h"

namespace vllm_ascend
{
    namespace npu_kernel
    {
        std::tuple<at::Tensor, at::Tensor>
        get_masked_input_and_mask(at::Tensor&   input,
                                  const int64_t org_vocab_start_index,
                                  const int64_t org_vocab_end_index,
                                  const int64_t num_org_vocab_padding,
                                  const int64_t added_vocab_start_index,
                                  const int64_t added_vocab_end_index)
    /*
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py#L161-L198
    Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately (both padded),
    and place them in the same tensor.
    In this example, we will have the original vocab size = 1010,
    added vocab size = 16 and padding to 64. Therefore, the total
    vocab size with padding will be 1088 (because we first pad 1010 to
    1024, add 16, and then pad to 1088).
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< --------BASE-------- >|< -BASE PADDING-- >|< -----LORA------ >|< -LORA PADDING-- >|
    corresponding token_id: |  0  |  1  | ... | 1009 |  -1  | ... |  -1  | 1010 | ... | 1015 |  -1  | ... |  -1  |
                     index: |  0  |  1  | ... | 1009 | 1010 | ... | 1023 | 1024 | ... | 1039 | 1040 | ... | 1087 |

    TP2, rank 0:
                            |< --------------------BASE--------------------- >|< -----LORA------ >|< -LORA PADDING- >|
    corresponding token_id: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 1000 | ... | 1015 |  -1  | ... |  -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  |  520 | ... | 543 |
    TP2, rank 1:
                            |< -----------BASE----------- >|< -BASE PADDING- >|< -----------LORA PADDING----------- >|
    corresponding token_id: | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |  -1  | ... |  -1  | -1  | ... |   -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 519  | 520 | ... |  543 |
    Parameters:
        org_vocab_start_index //base embeddings start
        org_vocab_end_index //base embeddings end
        num_org_vocab_padding //base embeddings padding
        added_vocab_start_index //LoRA embeddings start
        added_vocab_end_index //LoRA embeddings end
    */
{
    // Input validation
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    TORCH_CHECK(org_vocab_start_index >= 0, "org_vocab_start_index must be non-negative");
    TORCH_CHECK(org_vocab_end_index >= org_vocab_start_index,
                "org_vocab_end_index must be greater than org_vocab_start_index");
    TORCH_CHECK(num_org_vocab_padding >= 0, "num_org_vocab_padding must be non-negative");
    TORCH_CHECK(added_vocab_start_index >= org_vocab_end_index,
                "added_vocab_start_index must be greater than org_vocab_end_index");
    TORCH_CHECK(added_vocab_end_index >= added_vocab_start_index,
                "added_vocab_end_index must be greater than added_vocab_start_index");

    // Get total number of elements
    int64_t size = input.numel();

    // Create output tensors
    at::Tensor masked_input = at::empty_like(input);
    at::Tensor mask         = at::empty_like(input).to(at::kBool);

    // Get data pointers
    void* input_ptr        = input.data_ptr();
    void* masked_input_ptr = masked_input.data_ptr();
    void* mask_ptr         = mask.data_ptr();

    // Get current stream
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    // Get scalar type
    at::ScalarType scalar_type = input.scalar_type();

    // Create and configure OpCommand
    at_npu::native::OpCommand cmd;
    cmd.Name("get_masked_input_and_mask");
    cmd
            .SetCustomHandler(
                    [scalar_type,
                     size,
                     stream,
                     input_ptr,
                     masked_input_ptr,
                     mask_ptr,
                     org_vocab_start_index,
                     org_vocab_end_index,
                     num_org_vocab_padding,
                     added_vocab_start_index,
                     added_vocab_end_index]() -> int
                    {
                        int     device_id = 0;
                        int64_t aiv_num   = 0;
                        TORCH_CHECK(aclGetDeviceCapability(device_id,
                                                           ACL_DEVICE_INFO_VECTOR_CORE_NUM,
                                                           &aiv_num) == ACL_SUCCESS);
                        uint32_t loop_cnt = (size + aiv_num - 1) / aiv_num;

                        // Call implementation
                        get_masked_input_and_mask_impl(stream,
                                                       input_ptr,
                                                       masked_input_ptr,
                                                       mask_ptr,
                                                       org_vocab_start_index,
                                                       org_vocab_end_index,
                                                       num_org_vocab_padding,
                                                       added_vocab_start_index,
                                                       added_vocab_end_index,
                                                       size,
                                                       loop_cnt,
                                                       aiv_num);

                        return 0;
                    });
    cmd.Run();
    return {masked_input, mask};
        }
    }  // namespace npu_kernel

    namespace meta
    {
        std::tuple<at::Tensor, at::Tensor>
        get_masked_input_and_mask_meta(at::Tensor&   input,
                                       const int64_t org_vocab_start_index,
                                       const int64_t org_vocab_end_index,
                                       const int64_t num_org_vocab_padding,
                                       const int64_t added_vocab_start_index,
                                       const int64_t added_vocab_end_index)
        {
            at::Tensor masked_input = at::empty_like(input);
            at::Tensor mask         = at::empty_like(input, input.options().dtype(at::kBool));

            return {masked_input, mask};
        }
    }  // namespace meta
}  // namespace vllm_ascend
