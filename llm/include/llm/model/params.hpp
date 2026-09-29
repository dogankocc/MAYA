#pragma once

#include <random>

#include "llm/tensor/tensor.hpp"

namespace llm::model {

void InitTensorXavier(Tensor& tensor, std::mt19937& rng);

// GPT-2 / LLaMA / nanoGPT standard: N(0, stddev)
void InitTensorNormal(Tensor& tensor, std::mt19937& rng, float stddev = 0.02f);

void InitTensorZeros(Tensor& tensor);

} // namespace llm::model
