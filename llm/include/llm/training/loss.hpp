#pragma once

#include <vector>

#include "llm/core/types.hpp"
#include "llm/tensor/tensor.hpp"

namespace llm::training {

[[nodiscard]] float CrossEntropyLoss(const Tensor& logits, const std::vector<TokenId>& targetTokens, Tensor& gradLogits);

} // namespace llm::training
