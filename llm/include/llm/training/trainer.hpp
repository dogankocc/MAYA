#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/core/types.hpp"
#include "llm/model/transformer.hpp"
#include "llm/training/adamw.hpp"
#include "llm/training/parameter.hpp"

namespace llm::training {

struct TrainerConfig {
  AdamWConfig optimizer;
  std::size_t maxSeqLen = 0;
};

class Trainer {
public:
  Trainer(model::TransformerModel& model, TrainerConfig config);

  [[nodiscard]] Status TrainStep(const std::vector<TokenId>& tokens, float& loss);

  [[nodiscard]] Status TrainEpoch(const std::vector<std::vector<TokenId>>& batches, float& averageLoss);

  [[nodiscard]] ParameterList& Parameters() { return parameters_; }

  [[nodiscard]] AdamW& Optimizer() { return optimizer_; }

private:
  model::TransformerModel& model_;
  TrainerConfig config_;
  ParameterList parameters_;
  AdamW optimizer_;
};

} // namespace llm::training
