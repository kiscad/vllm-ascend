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

#include "utils/types.h"

namespace vllm_ascend {
namespace npu_kernel {

    extern void bgmv_expand_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *indices,
        uint32_t indicesSize,
        void *y,
        void *y_out,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t lora_rank,
        uint32_t output_hidden_dim,
        uint32_t slice_offset,
        uint32_t output_full_dim
    );

} // namespace npu_kernel
} // namespace vllm_ascend
