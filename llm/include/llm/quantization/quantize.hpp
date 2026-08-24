#pragma once

#include <cstdint>
#include <vector>

#include "llm/tensor/tensor.hpp"

namespace llm::quantization {

struct QuantizedTensor {
  Shape shape;
  float scale = 1.0f;
  std::vector<std::int8_t> data;
};

[[nodiscard]] QuantizedTensor QuantizeSymmetric(const Tensor& tensor);

[[nodiscard]] Tensor Dequantize(const QuantizedTensor& quantized);

[[nodiscard]] float MaxAbsError(const Tensor& a, const Tensor& b);

} // namespace llm::quantization
