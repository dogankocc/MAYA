#include <gtest/gtest.h>
#include <filesystem>
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

TEST(AdamWStateTest, SaveLoadReproducesUninterruptedTraining) {
  const llm::ModelConfig config = TinyConfig();
  const std::vector<llm::TokenId> sample = {3, 4, 5, 6, 7};
  const std::string statePath =
      (std::filesystem::temp_directory_path() / "maya_adamw_state_test.bin").string();

  auto makeModel = [&] {
    llm::model::TransformerModel model(config);
    std::mt19937 rng(11);
    model.ResetParameters(rng);
    return model;
  };
  llm::training::TrainerConfig trainerConfig;
  trainerConfig.optimizer.learningRate = 0.05f;

  llm::model::TransformerModel continuous = makeModel();
  llm::training::Trainer continuousTrainer(continuous, trainerConfig);
  float loss = 0.0f;
  for (int step = 0; step < 4; ++step) {
    ASSERT_TRUE(continuousTrainer.TrainStep(sample, loss).IsOk());
  }
  ASSERT_TRUE(continuousTrainer.Optimizer().SaveState(statePath).IsOk());
  EXPECT_EQ(continuousTrainer.Optimizer().StepCount(), 4u);

  llm::model::TransformerModel resumed = makeModel();
  llm::training::Trainer resumedTrainer(resumed, trainerConfig);
  for (int step = 0; step < 4; ++step) {
    ASSERT_TRUE(resumedTrainer.TrainStep(sample, loss).IsOk());
  }
  resumedTrainer.Optimizer().Reset();
  ASSERT_TRUE(resumedTrainer.Optimizer().LoadState(statePath, resumedTrainer.Parameters()).IsOk());
  EXPECT_EQ(resumedTrainer.Optimizer().StepCount(), 4u);

  float continuousLoss = 0.0f;
  float resumedLoss = 0.0f;
  ASSERT_TRUE(continuousTrainer.TrainStep(sample, continuousLoss).IsOk());
  ASSERT_TRUE(resumedTrainer.TrainStep(sample, resumedLoss).IsOk());
  EXPECT_FLOAT_EQ(continuousLoss, resumedLoss);

  const llm::Tensor& lhs = continuous.TokenEmbedding();
  const llm::Tensor& rhs = resumed.TokenEmbedding();
  for (llm::Index i = 0; i < static_cast<llm::Index>(lhs.Numel()); ++i) {
    ASSERT_FLOAT_EQ(lhs[i], rhs[i]);
  }

  llm::model::TransformerModel other(config);
  std::mt19937 otherRng(1);
  other.ResetParameters(otherRng);
  llm::training::Trainer otherTrainer(other, trainerConfig);
  EXPECT_TRUE(otherTrainer.Optimizer().LoadState(statePath, otherTrainer.Parameters()).IsOk());

  std::filesystem::remove(statePath);
}
