#include <gtest/gtest.h>
#include <random>

#include "llm/inference/continuous_batch_engine.hpp"
#include "llm/inference/sampling/sampler.hpp"
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

llm::TokenId GreedyRow(const llm::Tensor& logits, const std::size_t row) {
  llm::Tensor rowTensor = llm::Tensor::Zeros(llm::Shape{1, logits.GetShape()[1]});
  for (llm::Dimension vocab = 0; vocab < logits.GetShape()[1]; ++vocab) {
    rowTensor.At({0, static_cast<llm::Index>(vocab)}) =
        logits.At({static_cast<llm::Index>(row), static_cast<llm::Index>(vocab)});
  }
  return llm::inference::Sampler::SampleGreedy(rowTensor);
}

} // namespace

TEST(ContinuousBatchEngineTest, QueuesWhenSlotsFull) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(37);
  model.ResetParameters(rng);

  llm::inference::ContinuousBatchEngine engine(model, 2);

  std::size_t slotId = 0;
  llm::Tensor prefillLogits = llm::Tensor::Zeros(llm::Shape{1, config.vocabSize});
  ASSERT_TRUE(engine.Enqueue({3, 4}, slotId, prefillLogits).IsOk());
  EXPECT_LT(slotId, 2U);
  ASSERT_TRUE(engine.Enqueue({5, 6, 7}, slotId, prefillLogits).IsOk());
  EXPECT_LT(slotId, 2U);
  ASSERT_TRUE(engine.Enqueue({8, 9}, slotId, prefillLogits).IsOk());
  EXPECT_EQ(slotId, 2U);
  EXPECT_EQ(engine.PendingCount(), 1U);
}

TEST(ContinuousBatchEngineTest, AdmitsPendingAfterSlotFrees) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(41);
  model.ResetParameters(rng);

  llm::inference::ContinuousBatchEngine engine(model, 2);

  std::size_t slotId = 0;
  llm::Tensor prefillLogits = llm::Tensor::Zeros(llm::Shape{1, config.vocabSize});
  ASSERT_TRUE(engine.Enqueue({3, 4}, slotId, prefillLogits).IsOk());
  ASSERT_TRUE(engine.Enqueue({5, 6}, slotId, prefillLogits).IsOk());
  ASSERT_TRUE(engine.Enqueue({7, 8, 9}, slotId, prefillLogits).IsOk());
  EXPECT_EQ(engine.PendingCount(), 1U);

  std::vector<llm::TokenId> nextTokens(2, 0);
  nextTokens[0] = GreedyRow(prefillLogits, 0);
  nextTokens[1] = GreedyRow(prefillLogits, 0);

  llm::Tensor logits = llm::Tensor::Zeros(llm::Shape{2, config.vocabSize});
  std::vector<bool> finished(2, false);

  for (int step = 0; step < static_cast<int>(config.maxSeqLen); ++step) {
    for (std::size_t slot = 0; slot < 2; ++slot) {
      if (engine.Slots()[slot].state == llm::inference::SlotState::Decoding) {
        nextTokens[slot] = GreedyRow(logits, slot);
      }
    }

    ASSERT_TRUE(engine.Step(nextTokens, logits, finished).IsOk());

    if (engine.PendingCount() == 0) {
      bool hasDecoding = false;
      for (const llm::inference::BatchSlot& slot : engine.Slots()) {
        if (slot.state == llm::inference::SlotState::Decoding) {
          hasDecoding = true;
          break;
        }
      }
      if (!hasDecoding) {
        break;
      }
    }
  }

  EXPECT_EQ(engine.PendingCount(), 0U);
}
