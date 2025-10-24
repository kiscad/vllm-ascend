#pragma once

#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <torch/version.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>

#include <tuple>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "types.h"

namespace vllm_ascend
{
    AscendType get_dtype_from_torch(c10::ScalarType scalarType);
}
