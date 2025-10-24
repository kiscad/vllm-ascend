#pragma once

#include "../../utils/types.h"

namespace vllm_ascend
{
    namespace npu_kernel
    {

        extern void rotary_embedding_impl(AscendType     type,
                                          bool           isNeox,
                                          void*          stream,
                                          int64_t*       positions,
                                          void*          queryDst,
                                          void*          keyDst,
                                          void*          query,
                                          void*          key,
                                          void*          cosSinCache,
                                          const int      rotDim,
                                          const int64_t  queryStride,
                                          const int64_t  keyStride,
                                          const int64_t  dstQueryStride,
                                          const int64_t  dstKeyStride,
                                          const int      numHeads,
                                          const int      numKvHeads,
                                          const int      headSize,
                                          const int64_t  numTokens,
                                          const uint32_t loopCnt,
                                          uint32_t       aivNum);

    }  // namespace npu_kernel
}  // namespace vllm_ascend
