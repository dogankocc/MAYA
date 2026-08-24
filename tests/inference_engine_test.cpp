#include <cmath>
#include <gtest/gtest.h>
#include <random>

#include "llm/core/config.hpp"
#include "llm/inference/inference_engine.hpp"
#include "llm/model/transformer.hpp"

namespace {

llm::ModelConfig SmallConfig() {
  llm::ModelConfig config;
  config.vocabSize = 64;
  config.hiddenDim = 32;
  config.numLayers = 2;
  config.numHeads = 4;
  config.numKvHeads = 2;
  config.intermediateDim = 64;
  config.maxSeqLen = 32;
  return config;
}

} // namespace

TEST(InferenceEngineTest, PrefillMatchesFullForwardLastToken) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(42);
  model.ResetParameters(rng);

  const std::vector<llm::TokenId> tokens = {5, 6, 7, 8};

  llm::Tensor fullLogits = llm::Tensor::Zeros(llm::Shape{tokens.size(), config.vocabSize});
  ASSERT_TRUE(model.Forward(tokens, fullLogits).IsOk());

  llm::inference::InferenceEngine engine(model);
  llm::Tensor prefillLogits = llm::Tensor::Zeros(llm::Shape{1, config.vocabSize});
  ASSERT_TRUE(engine.Prefill(tokens, prefillLogits).IsOk());

  for (llm::Dimension vocab = 0; vocab < config.vocabSize; ++vocab) {
    EXPECT_NEAR(prefillLogits.At({0, static_cast<llm::Index>(vocab)}),
                fullLogits.At({3, static_cast<llm::Index>(vocab)}), 1e-4f);
  }
}

TEST(InferenceEngineTest, DecodeMatchesFullForward) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(99);
  model.ResetParameters(rng);

  const std::vector<llm::TokenId> prompt = {3, 4, 5};
  const llm::TokenId nextToken = 6;
  const std::vector<llm::TokenId> full = {3, 4, 5, 6};

  llm::Tensor fullLogits = llm::Tensor::Zeros(llm::Shape{full.size(), config.vocabSize});
  ASSERT_TRUE(model.Forward(full, fullLogits).IsOk());

  llm::inference::InferenceEngine engine(model);
  llm::Tensor prefillLogits = llm::Tensor::Zeros(llm::Shape{1, config.vocabSize});
  ASSERT_TRUE(engine.Prefill(prompt, prefillLogits).IsOk());
  EXPECT_EQ(engine.SequenceLength(), prompt.size());

  llm::Tensor decodeLogits = llm::Tensor::Zeros(llm::Shape{1, config.vocabSize});
  ASSERT_TRUE(engine.Decode(nextToken, decodeLogits).IsOk());
  EXPECT_EQ(engine.SequenceLength(), full.size());

  for (llm::Dimension vocab = 0; vocab < config.vocabSize; ++vocab) {
    EXPECT_NEAR(decodeLogits.At({0, static_cast<llm::Index>(vocab)}),
                fullLogits.At({3, static_cast<llm::Index>(vocab)}), 1e-4f);
  }
}

TEST(InferenceEngineTest, ResetClearsCacheLength) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(7);
  model.ResetParameters(rng);

  llm::inference::InferenceEngine engine(model);
  llm::Tensor logits = llm::Tensor::Zeros(llm::Shape{1, config.vocabSize});
  ASSERT_TRUE(engine.Prefill({1, 2}, logits).IsOk());
  EXPECT_EQ(engine.SequenceLength(), 2U);

  engine.Reset();
  EXPECT_EQ(engine.SequenceLength(), 0U);
}
