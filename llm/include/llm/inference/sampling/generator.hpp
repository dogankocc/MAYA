#pragma once

#include <random>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/core/types.hpp"
#include "llm/inference/inference_engine.hpp"
#include "llm/inference/sampling/sampler.hpp"

namespace llm::inference {

class TextGenerator {
public:
  TextGenerator(InferenceEngine& engine, SamplingConfig config);

  [[nodiscard]] Result<std::vector<TokenId>> Generate(const std::vector<TokenId>& prompt, std::size_t maxNewTokens,
                                                      std::mt19937& rng);

  [[nodiscard]] const SamplingConfig& GetConfig() const { return config_; }

private:
  InferenceEngine& engine_;
  SamplingConfig config_;
};

} // namespace llm::inference
