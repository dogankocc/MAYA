#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/core/types.hpp"
#include "llm/inference/batch_inference_engine.hpp"
#include "llm/inference/sampling/sampler.hpp"

namespace llm::inference {

class BatchTextGenerator {
public:
  explicit BatchTextGenerator(BatchInferenceEngine& engine, SamplingConfig config = {});

  [[nodiscard]] Result<std::vector<std::vector<TokenId>>> GenerateBatch(
      const std::vector<std::vector<TokenId>>& prompts, std::size_t maxNewTokens, std::mt19937& rng);

  [[nodiscard]] SamplingConfig& Config() { return config_; }

  [[nodiscard]] const SamplingConfig& Config() const { return config_; }

private:
  BatchInferenceEngine& engine_;
  SamplingConfig config_;
};

} // namespace llm::inference
