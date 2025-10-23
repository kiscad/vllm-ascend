/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "../../utils/types.h"

namespace vllm_ascend {
namespace npu_kernel {

    extern void sgmv_shrink_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *loraIndices,
        uint32_t loraIndicesSize,
        void *seqLen,
        uint32_t seqLenSize,
        void *y,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t input_hidden_dim,
        uint32_t lora_rank,
        float scale
    );

} // namespace npu_kernel
} // namespace vllm_ascend
