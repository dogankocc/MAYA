#pragma once

#include "llm/core/status.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::training {

void Accumulate(Tensor& target, const Tensor& source);

[[nodiscard]] Status MatMulBackward(const Tensor& a, const Tensor& b, const Tensor& gradOutput, Tensor& gradA,
                                  Tensor& gradB);

[[nodiscard]] Status AddTo(Tensor& target, const Tensor& source);

[[nodiscard]] Status MulBackward(const Tensor& a, const Tensor& b, const Tensor& gradOutput, Tensor& gradA,
                               Tensor& gradB);

void SiluBackward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput);

[[nodiscard]] Status RmsNormBackward(const Tensor& input, const Tensor& weight, const Tensor& output,
                                     const Tensor& gradOutput, Scalar eps, Tensor& gradInput, Tensor& gradWeight);

[[nodiscard]] Status LinearBackward(const Tensor& input, const Tensor& weight, const Tensor& gradOutput,
                                    Tensor& gradInput, Tensor& gradWeight);

} // namespace llm::training
