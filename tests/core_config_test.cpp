#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

#include "llm/core/config.hpp"

TEST(ConfigTest, DefaultConfigIsValid) {
  const llm::ModelConfig config = llm::Config::DefaultModelConfig();
  EXPECT_TRUE(config.Validate().IsOk());
}

TEST(ConfigTest, InvalidHeadDivisorFailsValidation) {
  llm::ModelConfig config = llm::Config::DefaultModelConfig();
  config.hiddenDim = 513;
  EXPECT_FALSE(config.Validate().IsOk());
}

TEST(ConfigTest, RoundTripConfigFile) {
  const std::string path = "test_model.conf";
  const llm::ModelConfig original = llm::Config::DefaultModelConfig();

  ASSERT_TRUE(llm::Config::SaveToFile(original, path).IsOk());

  const auto loaded = llm::Config::LoadFromFile(path);
  ASSERT_TRUE(loaded.IsOk());

  const llm::ModelConfig& config = loaded.Value();
  EXPECT_EQ(config.vocabSize, original.vocabSize);
  EXPECT_EQ(config.hiddenDim, original.hiddenDim);
  EXPECT_EQ(config.numLayers, original.numLayers);
  EXPECT_EQ(config.numHeads, original.numHeads);

  std::remove(path.c_str());
}
