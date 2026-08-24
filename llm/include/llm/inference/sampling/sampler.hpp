#pragma once

#include <cstddef>
#include <random>

#include "llm/core/types.hpp"
#include "llm/tensor/tensor.hpp"
#include "llm/tokenizer/vocabulary.hpp"

namespace llm::inference {

struct SamplingConfig {
  float temperature = 1.0f;
  std::size_t topK = 0;
  float topP = 1.0f;
  bool greedy = false;
  TokenId eosTokenId = kEosTokenId;
  float minFirstTokenProbability = 0.0f;
};

class Sampler {
public:
  [[nodiscard]] static TokenId SampleGreedy(const Tensor& logits);

  [[nodiscard]] static TokenId Sample(const Tensor& logits, const SamplingConfig& config, std::mt19937& rng);

  [[nodiscard]] static TokenId SampleRow(const Tensor& logits, std::size_t row, const SamplingConfig& config,
                                         std::mt19937& rng);

  [[nodiscard]] static float MaxTokenProbability(const Tensor& logits);
};

} // namespace llm::inference
