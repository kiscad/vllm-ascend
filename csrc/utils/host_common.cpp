#include <c10/core/ScalarType.h>
#include "types.h"

namespace vllm_ascend {
AscendType get_dtype_from_torch(c10::ScalarType scalarType)
{
    if (scalarType == c10::ScalarType::Float) {
        return AscendType::FP32;
    } else if (scalarType == c10::ScalarType::BFloat16) {
        return AscendType::BF16;
    } else {
        return AscendType::FP16;
    }
}
}
