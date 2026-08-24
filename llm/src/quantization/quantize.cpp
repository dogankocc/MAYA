#include "llm/quantization/quantize.hpp"

#include <algorithm>
#include <cmath>

namespace llm::quantization {

QuantizedTensor QuantizeSymmetric(const Tensor& tensor) {
  QuantizedTensor quantized;
  quantized.shape = tensor.GetShape();
  quantized.data.resize(tensor.Numel());

  float maxAbs = 0.0f;
  for (Index i = 0; i < static_cast<Index>(tensor.Numel()); ++i) {
    maxAbs = std::max(maxAbs, std::fabs(tensor[i]));
  }

  if (maxAbs < 1e-8f) {
    quantized.scale = 1.0f;
    std::fill(quantized.data.begin(), quantized.data.end(), static_cast<std::int8_t>(0));
    return quantized;
  }

  quantized.scale = maxAbs / 127.0f;
  for (Index i = 0; i < static_cast<Index>(tensor.Numel()); ++i) {
    const float scaled = tensor[i] / quantized.scale;
    const auto rounded = static_cast<long>(std::lround(scaled));
    const auto clamped = std::clamp(rounded, -127L, 127L);
    quantized.data[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(clamped);
  }

  return quantized;
}

Tensor Dequantize(const QuantizedTensor& quantized) {
  std::vector<Scalar> data(quantized.data.size());
  for (std::size_t i = 0; i < quantized.data.size(); ++i) {
    data[i] = static_cast<Scalar>(quantized.data[i]) * quantized.scale;
  }
  return Tensor::FromBuffer(quantized.shape, std::move(data));
}

float MaxAbsError(const Tensor& a, const Tensor& b) {
  float maxError = 0.0f;
  const Index count = static_cast<Index>(std::min(a.Numel(), b.Numel()));
  for (Index i = 0; i < count; ++i) {
    maxError = std::max(maxError, std::fabs(a[i] - b[i]));
  }
  return maxError;
}

} // namespace llm::quantization
