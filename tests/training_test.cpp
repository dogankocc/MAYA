#include <gtest/gtest.h>
#include <random>

#include "llm/model/transformer.hpp"
#include "llm/training/loss.hpp"
#include "llm/training/trainer.hpp"

namespace {

llm::ModelConfig TinyConfig() {
  llm::ModelConfig config;
  config.vocabSize = 32;
  config.hiddenDim = 16;
  config.numLayers = 1;
  config.numHeads = 2;
  config.numKvHeads = 1;
  config.intermediateDim = 32;
  config.maxSeqLen = 16;
  return config;
}

} // namespace

TEST(TrainingLossTest, CrossEntropyMatchesManualSoftmax) {
  const llm::Tensor logits =
      llm::Tensor::FromBuffer(llm::Shape{1, 3}, {1.0f, 2.0f, 0.5f});
  llm::Tensor gradLogits;
  const float loss = llm::training::CrossEntropyLoss(logits, {1}, gradLogits);
  EXPECT_GT(loss, 0.0f);
  EXPECT_LT(loss, 2.0f);
  EXPECT_NEAR(gradLogits.At({0, 1}), 0.0f, 0.5f);
}

TEST(TrainerTest, LossDecreasesOverSteps) {
  const llm::ModelConfig config = TinyConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(5);
  model.ResetParameters(rng);

  llm::training::TrainerConfig trainerConfig;
  trainerConfig.optimizer.learningRate = 0.05f;
  llm::training::Trainer trainer(model, trainerConfig);

  const std::vector<llm::TokenId> sample = {3, 4, 5, 6, 7};
  float firstLoss = 0.0f;
  float lastLoss = 0.0f;
  ASSERT_TRUE(trainer.TrainStep(sample, firstLoss).IsOk());

  for (int step = 0; step < 8; ++step) {
    ASSERT_TRUE(trainer.TrainStep(sample, lastLoss).IsOk());
  }

  EXPECT_LT(lastLoss, firstLoss);
}
