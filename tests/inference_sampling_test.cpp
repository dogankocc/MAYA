#include <cmath>
#include <gtest/gtest.h>
#include <random>

#include "llm/inference/sampling/generator.hpp"
#include "llm/inference/sampling/sampler.hpp"
#include "llm/model/transformer.hpp"

namespace {

llm::ModelConfig SmallConfig() {
  llm::ModelConfig config;
  config.vocabSize = 16;
  config.hiddenDim = 32;
  config.numLayers = 1;
  config.numHeads = 4;
  config.numKvHeads = 2;
  config.intermediateDim = 64;
  config.maxSeqLen = 32;
  return config;
}

llm::Tensor MakeLogits(const std::vector<llm::Scalar>& values) {
  return llm::Tensor::FromBuffer(llm::Shape{1, values.size()}, values);
}

} // namespace

TEST(SamplerTest, GreedySelectsArgmax) {
  const llm::Tensor logits = MakeLogits({1.0f, 3.0f, 2.0f});
  EXPECT_EQ(llm::inference::Sampler::SampleGreedy(logits), 1U);
}

TEST(SamplerTest, GreedyConfigMatchesArgmax) {
  const llm::Tensor logits = MakeLogits({0.5f, 0.1f, 0.9f});
  llm::inference::SamplingConfig config;
  config.greedy = true;

  std::mt19937 rng(1);
  EXPECT_EQ(llm::inference::Sampler::Sample(logits, config, rng), 2U);
}

TEST(SamplerTest, TopKLimitsCandidates) {
  const llm::Tensor logits = MakeLogits({10.0f, 9.0f, 1.0f, 0.5f});
  llm::inference::SamplingConfig config;
  config.temperature = 1.0f;
  config.topK = 2;

  std::mt19937 rng(123);
  for (int i = 0; i < 20; ++i) {
    const llm::TokenId token = llm::inference::Sampler::Sample(logits, config, rng);
    EXPECT_LT(token, 2U);
  }
}

TEST(TextGeneratorTest, GeneratesNewTokens) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(42);
  model.ResetParameters(rng);

  llm::inference::InferenceEngine engine(model);
  llm::inference::SamplingConfig sampling;
  sampling.greedy = true;
  sampling.eosTokenId = 999;

  llm::inference::TextGenerator generator(engine, sampling);
  const auto result = generator.Generate({1, 2}, 3, rng);
  ASSERT_TRUE(result.IsOk());
  EXPECT_EQ(result.Value().size(), 5U);
}

TEST(TextGeneratorTest, StopsAtEos) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(7);
  model.ResetParameters(rng);

  llm::inference::InferenceEngine engine(model);
  llm::Tensor prefillLogits = llm::Tensor::Zeros(llm::Shape{1, config.vocabSize});
  ASSERT_TRUE(engine.Prefill({3, 4}, prefillLogits).IsOk());

  llm::inference::SamplingConfig sampling;
  sampling.greedy = true;
  sampling.eosTokenId = llm::inference::Sampler::SampleGreedy(prefillLogits);

  llm::inference::TextGenerator generator(engine, sampling);
  const auto result = generator.Generate({3, 4}, 10, rng);
  ASSERT_TRUE(result.IsOk());
  EXPECT_EQ(result.Value().size(), 3U);
}
