#include <cmath>
#include <gtest/gtest.h>
#include <random>

#include "llm/inference/batch_inference_engine.hpp"
#include "llm/inference/sampling/batch_generator.hpp"
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

TEST(BatchInferenceEngineTest, PrefillBatchMatchesSingleForward) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(11);
  model.ResetParameters(rng);

  const std::vector<std::vector<llm::TokenId>> prompts = {{5, 6, 7}, {3, 4, 5, 6}};

  llm::inference::BatchInferenceEngine engine(model, prompts.size());
  llm::Tensor batchLogits = llm::Tensor::Zeros(llm::Shape{prompts.size(), config.vocabSize});
  ASSERT_TRUE(engine.PrefillBatch(prompts, batchLogits).IsOk());

  for (std::size_t batchIndex = 0; batchIndex < prompts.size(); ++batchIndex) {
    llm::Tensor fullLogits = llm::Tensor::Zeros(llm::Shape{prompts[batchIndex].size(), config.vocabSize});
    ASSERT_TRUE(model.Forward(prompts[batchIndex], fullLogits).IsOk());

    const llm::Dimension lastIndex = static_cast<llm::Dimension>(prompts[batchIndex].size() - 1);
    for (llm::Dimension vocab = 0; vocab < config.vocabSize; ++vocab) {
      EXPECT_NEAR(batchLogits.At({static_cast<llm::Index>(batchIndex), static_cast<llm::Index>(vocab)}),
                  fullLogits.At({static_cast<llm::Index>(lastIndex), static_cast<llm::Index>(vocab)}), 1e-4f);
    }
  }
}

TEST(BatchInferenceEngineTest, DecodeBatchMatchesSingleForward) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(17);
  model.ResetParameters(rng);

  const std::vector<std::vector<llm::TokenId>> prompts = {{3, 4, 5}, {7, 8, 9, 10}};
  const std::vector<llm::TokenId> nextTokens = {6, 11};

  llm::inference::BatchInferenceEngine engine(model, prompts.size());
  llm::Tensor prefillLogits = llm::Tensor::Zeros(llm::Shape{prompts.size(), config.vocabSize});
  ASSERT_TRUE(engine.PrefillBatch(prompts, prefillLogits).IsOk());

  llm::Tensor decodeLogits = llm::Tensor::Zeros(llm::Shape{prompts.size(), config.vocabSize});
  ASSERT_TRUE(engine.DecodeBatch(nextTokens, decodeLogits).IsOk());

  for (std::size_t batchIndex = 0; batchIndex < prompts.size(); ++batchIndex) {
    std::vector<llm::TokenId> full = prompts[batchIndex];
    full.push_back(nextTokens[batchIndex]);

    llm::Tensor fullLogits = llm::Tensor::Zeros(llm::Shape{full.size(), config.vocabSize});
    ASSERT_TRUE(model.Forward(full, fullLogits).IsOk());

    const llm::Dimension lastIndex = static_cast<llm::Dimension>(full.size() - 1);
    for (llm::Dimension vocab = 0; vocab < config.vocabSize; ++vocab) {
      EXPECT_NEAR(decodeLogits.At({static_cast<llm::Index>(batchIndex), static_cast<llm::Index>(vocab)}),
                  fullLogits.At({static_cast<llm::Index>(lastIndex), static_cast<llm::Index>(vocab)}), 1e-4f);
    }
  }
}

TEST(BatchInferenceEngineTest, PaddedPrefillMatchesSlotPrefill) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(29);
  model.ResetParameters(rng);

  const std::vector<std::vector<llm::TokenId>> prompts = {{5, 6, 7}, {3, 4, 5, 6}};

  llm::inference::BatchInferenceEngine paddedEngine(model, prompts.size());
  paddedEngine.SetUsePaddedPrefill(true);
  llm::Tensor paddedLogits = llm::Tensor::Zeros(llm::Shape{prompts.size(), config.vocabSize});
  ASSERT_TRUE(paddedEngine.PrefillBatch(prompts, paddedLogits).IsOk());

  llm::inference::BatchInferenceEngine slotEngine(model, prompts.size());
  slotEngine.SetUsePaddedPrefill(false);
  llm::Tensor slotLogits = llm::Tensor::Zeros(llm::Shape{prompts.size(), config.vocabSize});
  ASSERT_TRUE(slotEngine.PrefillBatch(prompts, slotLogits).IsOk());

  for (llm::Index batchIndex = 0; batchIndex < static_cast<llm::Index>(prompts.size()); ++batchIndex) {
    for (llm::Dimension vocab = 0; vocab < config.vocabSize; ++vocab) {
      EXPECT_NEAR(paddedLogits.At({batchIndex, static_cast<llm::Index>(vocab)}),
                  slotLogits.At({batchIndex, static_cast<llm::Index>(vocab)}), 1e-4f);
    }
  }
}

TEST(BatchInferenceEngineTest, PrefillSlotPreservesOtherSlots) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(31);
  model.ResetParameters(rng);

  llm::inference::BatchInferenceEngine engine(model, 2);
  const std::vector<llm::TokenId> prompt0 = {3, 4, 5};
  const std::vector<llm::TokenId> prompt1 = {7, 8};

  llm::Tensor logits0 = llm::Tensor::Zeros(llm::Shape{1, config.vocabSize});
  llm::Tensor logits1 = llm::Tensor::Zeros(llm::Shape{1, config.vocabSize});
  ASSERT_TRUE(engine.PrefillSlot(0, prompt0, logits0).IsOk());
  ASSERT_TRUE(engine.PrefillSlot(1, prompt1, logits1).IsOk());

  const std::vector<llm::TokenId> nextTokens = {6, 9};
  llm::Tensor decodeLogits = llm::Tensor::Zeros(llm::Shape{2, config.vocabSize});
  ASSERT_TRUE(engine.DecodeBatch(nextTokens, decodeLogits).IsOk());

  for (std::size_t batchIndex = 0; batchIndex < 2; ++batchIndex) {
    std::vector<llm::TokenId> full = batchIndex == 0 ? prompt0 : prompt1;
    full.push_back(nextTokens[batchIndex]);

    llm::Tensor fullLogits = llm::Tensor::Zeros(llm::Shape{full.size(), config.vocabSize});
    ASSERT_TRUE(model.Forward(full, fullLogits).IsOk());

    const llm::Dimension lastIndex = static_cast<llm::Dimension>(full.size() - 1);
    for (llm::Dimension vocab = 0; vocab < config.vocabSize; ++vocab) {
      EXPECT_NEAR(decodeLogits.At({static_cast<llm::Index>(batchIndex), static_cast<llm::Index>(vocab)}),
                  fullLogits.At({static_cast<llm::Index>(lastIndex), static_cast<llm::Index>(vocab)}), 1e-4f);
    }
  }
}

TEST(BatchTextGeneratorTest, GeneratesForAllPrompts) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(23);
  model.ResetParameters(rng);

  const std::vector<std::vector<llm::TokenId>> prompts = {{3, 4}, {5, 6, 7}};

  llm::inference::BatchInferenceEngine engine(model, prompts.size());
  llm::inference::BatchTextGenerator generator(engine);
  generator.Config().greedy = true;

  const auto sequences = generator.GenerateBatch(prompts, 3, rng);
  ASSERT_TRUE(sequences.IsOk());
  ASSERT_EQ(sequences.Value().size(), prompts.size());

  for (std::size_t batchIndex = 0; batchIndex < prompts.size(); ++batchIndex) {
    EXPECT_GE(sequences.Value()[batchIndex].size(), prompts[batchIndex].size() + 1);
  }
}
