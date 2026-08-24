#pragma once

#include "llm/core/types.hpp"

namespace llm::tensor::detail {

void MatMulAvx2(const Scalar* a, const Scalar* b, Scalar* c, Dimension m, Dimension k, Dimension n);

void MatMulScalar(const Scalar* a, const Scalar* b, Scalar* c, Dimension m, Dimension k, Dimension n);

void MatMulQuantizedScalar(const Scalar* input, const std::int8_t* weight, Scalar scale, Scalar* output, Dimension m,
                           Dimension k, Dimension n);

void MatMulQuantizedAvx2(const Scalar* input, const std::int8_t* weight, Scalar scale, Scalar* output, Dimension m,
                         Dimension k, Dimension n);

} // namespace llm::tensor::detail
