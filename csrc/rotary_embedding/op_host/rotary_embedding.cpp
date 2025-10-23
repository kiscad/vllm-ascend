#include "../op_kernel/rotary_embedding_kernel.h"
#include "utils/host_common.h"

namespace vllm_ascend {
namespace npu_kernel {
std::tuple<at::Tensor, at::Tensor> rotary_embedding(at::Tensor &positions, at::Tensor &query, at::Tensor &key,
    int64_t head_size, at::Tensor &cos_sin_cache,  bool is_neox)
{
    int32_t deviceId = 0;
    int64_t num_tokens = positions.numel();
    int positions_ndim = positions.dim();
    TORCH_CHECK(
        positions_ndim == 1 || positions_ndim == 2,
        "positions must have shape [num_tokens] or [batch_size, seq_len]");
    if (positions_ndim == 1) {
      TORCH_CHECK(
          query.size(0) == positions.size(0) && key.size(0) == positions.size(0),
          "query, key and positions must have the same number of tokens");
    }
    if (positions_ndim == 2) {
      TORCH_CHECK(
          query.size(0) == positions.size(0) &&
              key.size(0) == positions.size(0) &&
              query.size(1) == positions.size(1) &&
              key.size(1) == positions.size(1),
          "query, key and positions must have the same batch_size and seq_len");
    }
    TORCH_CHECK(head_size % 32 == 0, "rotary_embedding: headSize should be divisible by 32");
    int query_hidden_size = query.numel() / num_tokens;
    int key_hidden_size = key.numel() / num_tokens;
    TORCH_CHECK(query_hidden_size % head_size == 0);
    TORCH_CHECK(key_hidden_size % head_size == 0);
    TORCH_CHECK(is_neox == true, "rotary_embedding: neox=false is not supported as custom kernel in vllm-ascend");

    // Make sure query and key have consistent number of heads
    int num_heads = query_hidden_size / head_size;
    int num_kv_heads = key_hidden_size / head_size;
    TORCH_CHECK(num_heads % num_kv_heads == 0);
    at::Tensor query_dst = at::empty({num_tokens, num_heads, head_size}, query.options());
    at::Tensor key_dst = at::empty({num_tokens, num_kv_heads, head_size}, key.options());

    int rot_dim = cos_sin_cache.size(1);
    int seq_dim_idx = positions_ndim - 1;
    int64_t *position_ids_ptr = positions.data_ptr<int64_t>();
    void *query_dst_ptr = query_dst.data_ptr();
    void *key_dst_ptr = key_dst.data_ptr();
    void *query_ptr = query.data_ptr();
    void *key_ptr = key.data_ptr();
    void *cos_sin_cache_ptr = cos_sin_cache.data_ptr();
    int64_t query_stride = query.stride(seq_dim_idx);
    int64_t key_stride = key.stride(seq_dim_idx);
    int64_t dst_query_stride = query_dst.stride(0);
    int64_t dst_key_stride = key_dst.stride(0);
    at::ScalarType scalar_type = query.scalar_type();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("rotary_embedding");
    cmd.SetCustomHandler([scalar_type, is_neox, num_tokens, stream, position_ids_ptr, query_dst_ptr, key_dst_ptr,
                          query_ptr, key_ptr, cos_sin_cache_ptr, rot_dim, query_stride, key_stride,
                          dst_query_stride, dst_key_stride, num_heads, num_kv_heads, head_size]() -> int {
        auto dtype_num = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        uint32_t loop_cnt = (num_tokens + aiv_num - 1) / aiv_num;
        rotary_embedding_impl(dtype_num, is_neox, stream, position_ids_ptr, query_dst_ptr, key_dst_ptr, query_ptr,
                                key_ptr, cos_sin_cache_ptr, rot_dim, query_stride, key_stride, dst_query_stride,
                                dst_key_stride, num_heads, num_kv_heads, head_size, num_tokens, loop_cnt, aiv_num);
        return 0;
    });
    cmd.Run();
    return {query_dst, key_dst};
}
} // namespace npu_kernel

namespace meta {
std::tuple<at::Tensor, at::Tensor> rotary_embedding_meta(
    at::Tensor &positions,
    at::Tensor &query,
    at::Tensor &key,
    int64_t head_size,
    at::Tensor &cos_sin_cache,
    bool is_neox)
{
    auto num_tokens = positions.sym_numel();
    auto query_hidden_size = query.sym_numel() / num_tokens;
    auto key_hidden_size = key.sym_numel() / num_tokens;

    auto num_heads = query_hidden_size / head_size;
    auto num_kv_heads = key_hidden_size / head_size;
    at::Tensor query_dst = at::empty_symint({num_tokens, num_heads, head_size}, query.options());
    at::Tensor key_dst = at::empty_symint({num_tokens, num_kv_heads, head_size}, key.options());

    return {query_dst, key_dst};
}
} // namespace meta
} // namespace vllm_ascend
