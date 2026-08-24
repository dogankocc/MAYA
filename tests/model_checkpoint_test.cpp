#include <cmath>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <random>

#include "llm/core/config.hpp"
#include "llm/model/checkpoint/checkpoint.hpp"
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

TEST(ModelCheckpointTest, SaveAndLoadPreservesForwardOutput) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(1234);
  model.ResetParameters(rng);

  const std::vector<llm::TokenId> tokens = {4, 5, 6};
  llm::Tensor logitsBefore = llm::Tensor::Zeros(llm::Shape{tokens.size(), config.vocabSize});
  ASSERT_TRUE(model.Forward(tokens, logitsBefore).IsOk());

  const std::string path = "test_model.ckpt";
  ASSERT_TRUE(llm::model::Checkpoint::Save(model, path).IsOk());

  const auto loaded = llm::model::Checkpoint::Load(path);
  ASSERT_TRUE(loaded.IsOk());

  llm::Tensor logitsAfter = llm::Tensor::Zeros(llm::Shape{tokens.size(), config.vocabSize});
  ASSERT_TRUE(loaded.Value().Forward(tokens, logitsAfter).IsOk());

  for (llm::Index i = 0; i < static_cast<llm::Index>(logitsBefore.Numel()); ++i) {
    EXPECT_NEAR(logitsBefore[i], logitsAfter[i], 1e-6f);
  }

  std::remove(path.c_str());
}

TEST(ModelCheckpointTest, RejectsInvalidMagic) {
  const std::string path = "bad_model.ckpt";
  {
    std::ofstream output(path, std::ios::binary);
    output.write("BADFILE1", 8);
  }

  const auto loaded = llm::model::Checkpoint::Load(path);
  EXPECT_FALSE(loaded.IsOk());
  std::remove(path.c_str());
}

TEST(ModelCheckpointTest, LoadedConfigMatchesOriginal) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(77);
  model.ResetParameters(rng);

  const std::string path = "test_model_config.ckpt";
  ASSERT_TRUE(llm::model::Checkpoint::Save(model, path).IsOk());

  const auto loaded = llm::model::Checkpoint::Load(path);
  ASSERT_TRUE(loaded.IsOk());

  const llm::ModelConfig& loadedConfig = loaded.Value().GetConfig();
  EXPECT_EQ(loadedConfig.vocabSize, config.vocabSize);
  EXPECT_EQ(loadedConfig.hiddenDim, config.hiddenDim);
  EXPECT_EQ(loadedConfig.numLayers, config.numLayers);
  EXPECT_EQ(loadedConfig.numHeads, config.numHeads);
  EXPECT_EQ(loadedConfig.numKvHeads, config.numKvHeads);

  std::remove(path.c_str());
}
