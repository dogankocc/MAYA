#pragma once

#include "llm/core/status.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::model {

[[nodiscard]] Status RmsNorm(const Tensor& input, const Tensor& weight, Scalar eps, Tensor& output);

} // namespace llm::model
