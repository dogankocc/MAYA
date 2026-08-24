#pragma once

#include <vector>

#include "llm/core/status.hpp"
#include "llm/core/types.hpp"
#include "llm/model/transformer.hpp"
#include "llm/training/parameter.hpp"

namespace llm::training {

[[nodiscard]] Status RunTrainBackward(model::TransformerModel& model, ParameterList& parameters,
                                      const std::vector<TokenId>& tokens, float& loss);

} // namespace llm::training
