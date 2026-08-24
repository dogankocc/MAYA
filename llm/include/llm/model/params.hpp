#pragma once

#include <random>

#include "llm/tensor/tensor.hpp"

namespace llm::model {

void InitTensorXavier(Tensor& tensor, std::mt19937& rng);

void InitTensorZeros(Tensor& tensor);

} // namespace llm::model
