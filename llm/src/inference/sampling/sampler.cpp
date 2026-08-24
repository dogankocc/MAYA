#include "llm/inference/sampling/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace llm::inference {

namespace {

[[nodiscard]] std::vector<Scalar> LogitsRow(const Tensor& logits) {
  const Dimension vocabSize = logits.GetShape()[1];
  std::vector<Scalar> row(static_cast<std::size_t>(vocabSize));
  for (Dimension index = 0; index < vocabSize; ++index) {
    row[static_cast<std::size_t>(index)] = logits.At({0, static_cast<Index>(index)});
  }
  return row;
}

[[nodiscard]] std::vector<Scalar> LogitsRowAt(const Tensor& logits, const std::size_t row) {
  const Dimension vocabSize = logits.GetShape()[1];
  std::vector<Scalar> values(static_cast<std::size_t>(vocabSize));
  for (Dimension index = 0; index < vocabSize; ++index) {
    values[static_cast<std::size_t>(index)] = logits.At({static_cast<Index>(row), static_cast<Index>(index)});
  }
  return values;
}

void SoftmaxInPlace(std::vector<Scalar>& values) {
  if (values.empty()) {
    return;
  }

  const Scalar maxValue = *std::max_element(values.begin(), values.end());
  Scalar sum = 0.0f;

  for (Scalar& value : values) {
    value = std::exp(value - maxValue);
    sum += value;
  }

  if (sum > 0.0f) {
    for (Scalar& value : values) {
      value /= sum;
    }
  }
}

void ApplyTemperature(std::vector<Scalar>& logits, const float temperature) {
  if (temperature <= 0.0f) {
    return;
  }

  for (Scalar& logit : logits) {
    logit /= temperature;
  }
}

void ApplyTopK(std::vector<Scalar>& logits, const std::size_t topK) {
  if (topK == 0 || topK >= logits.size()) {
    return;
  }

  std::vector<Scalar> sorted = logits;
  std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(topK - 1), sorted.end(),
                 std::greater<Scalar>());
  const Scalar threshold = sorted[topK - 1];

  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (logits[index] < threshold) {
      logits[index] = -std::numeric_limits<Scalar>::infinity();
    }
  }
}

void ApplyTopP(std::vector<Scalar>& probabilities, const float topP) {
  if (topP >= 1.0f) {
    return;
  }

  std::vector<std::pair<Scalar, std::size_t>> ranked;
  ranked.reserve(probabilities.size());
  for (std::size_t index = 0; index < probabilities.size(); ++index) {
    ranked.emplace_back(probabilities[index], index);
  }

  std::sort(ranked.begin(), ranked.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

  Scalar cumulative = 0.0f;
  std::vector<bool> keep(probabilities.size(), false);
  for (const auto& [probability, index] : ranked) {
    keep[index] = true;
    cumulative += probability;
    if (cumulative >= topP) {
      break;
    }
  }

  for (std::size_t index = 0; index < probabilities.size(); ++index) {
    if (!keep[index]) {
      probabilities[index] = 0.0f;
    }
  }

  const Scalar sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
  if (sum > 0.0f) {
    for (Scalar& probability : probabilities) {
      probability /= sum;
    }
  }
}

[[nodiscard]] TokenId SampleFromDistribution(const std::vector<Scalar>& probabilities, std::mt19937& rng) {
  std::uniform_real_distribution<Scalar> distribution(0.0f, 1.0f);
  const Scalar target = distribution(rng);
  Scalar cumulative = 0.0f;

  for (std::size_t index = 0; index < probabilities.size(); ++index) {
    cumulative += probabilities[index];
    if (target <= cumulative) {
      return static_cast<TokenId>(index);
    }
  }

  return static_cast<TokenId>(probabilities.size() - 1);
}

} // namespace

TokenId Sampler::SampleGreedy(const Tensor& logits) {
  const std::vector<Scalar> row = LogitsRow(logits);
  const auto best = std::max_element(row.begin(), row.end());
  return static_cast<TokenId>(std::distance(row.begin(), best));
}

TokenId Sampler::SampleRow(const Tensor& logits, const std::size_t row, const SamplingConfig& config,
                           std::mt19937& rng) {
  Tensor rowLogits = Tensor::Zeros(Shape{1, logits.GetShape()[1]});
  for (Dimension vocab = 0; vocab < logits.GetShape()[1]; ++vocab) {
    rowLogits.At({0, static_cast<Index>(vocab)}) = logits.At({static_cast<Index>(row), static_cast<Index>(vocab)});
  }
  return Sample(rowLogits, config, rng);
}

TokenId Sampler::Sample(const Tensor& logits, const SamplingConfig& config, std::mt19937& rng) {
  if (config.greedy || config.temperature <= 0.0f) {
    return SampleGreedy(logits);
  }

  std::vector<Scalar> row = LogitsRow(logits);
  ApplyTemperature(row, config.temperature);
  ApplyTopK(row, config.topK);
  SoftmaxInPlace(row);
  ApplyTopP(row, config.topP);

  return SampleFromDistribution(row, rng);
}

float Sampler::MaxTokenProbability(const Tensor& logits) {
  std::vector<Scalar> row = LogitsRow(logits);
  SoftmaxInPlace(row);
  return *std::max_element(row.begin(), row.end());
}

} // namespace llm::inference
