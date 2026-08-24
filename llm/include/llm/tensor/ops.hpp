#pragma once

#include "llm/core/status.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::quantization {
struct QuantizedTensor;
}

namespace llm::tensor {

[[nodiscard]] Status MatMul(const Tensor& a, const Tensor& b, Tensor& out);

[[nodiscard]] Status MatMulQuantized(const Tensor& input, const quantization::QuantizedTensor& weight, Tensor& out);

[[nodiscard]] Status Add(const Tensor& a, const Tensor& b, Tensor& out);

[[nodiscard]] Status Sub(const Tensor& a, const Tensor& b, Tensor& out);

[[nodiscard]] Status Mul(const Tensor& a, const Tensor& b, Tensor& out);

[[nodiscard]] Status Scale(const Tensor& input, Scalar factor, Tensor& out);

void Relu(Tensor& tensor);

void Gelu(Tensor& tensor);

void Silu(Tensor& tensor);

[[nodiscard]] Status Softmax(Tensor& tensor, Dimension axis);

} // namespace llm::tensor
