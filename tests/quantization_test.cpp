#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <random>

#include "llm/model/checkpoint/checkpoint.hpp"
#include "llm/model/transformer.hpp"
#include "llm/quantization/quant_checkpoint.hpp"
#include "llm/quantization/quantize.hpp"

namespace {

llm::ModelConfig SmallConfig() {
  llm::ModelConfig config;
  config.vocabSize = 32;
  config.hiddenDim = 32;
  config.numLayers = 1;
  config.numHeads = 4;
  config.numKvHeads = 2;
  config.intermediateDim = 64;
  config.maxSeqLen = 32;
  return config;
}

} // namespace

TEST(QuantizationTest, RoundTripLowError) {
  llm::Tensor tensor = llm::Tensor::FromBuffer(llm::Shape{4, 4}, {0.1f, -0.5f, 1.2f, -1.0f, 0.3f, 0.7f, -0.2f, 0.9f,
                                                                   0.4f, -0.8f, 0.6f, -0.1f, 0.2f, 0.5f, -0.4f, 0.8f});

  const llm::quantization::QuantizedTensor quantized = llm::quantization::QuantizeSymmetric(tensor);
  const llm::Tensor restored = llm::quantization::Dequantize(quantized);
  EXPECT_LT(llm::quantization::MaxAbsError(tensor, restored), 0.05f);
}

TEST(QuantizationTest, QuantCheckpointForwardCloseToFp32) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(42);
  model.ResetParameters(rng);

  const std::string fp32Path = "quant_fp32.ckpt";
  const std::string quantPath = "quant_int8.ckptq";
  ASSERT_TRUE(llm::model::Checkpoint::Save(model, fp32Path).IsOk());
  ASSERT_TRUE(llm::quantization::QuantCheckpoint::ConvertFile(fp32Path, quantPath).IsOk());

  const auto quantModel = llm::quantization::QuantCheckpoint::Load(quantPath);
  ASSERT_TRUE(quantModel.IsOk());

  const std::vector<llm::TokenId> tokens = {3, 4, 5};
  llm::Tensor fp32Logits = llm::Tensor::Zeros(llm::Shape{tokens.size(), config.vocabSize});
  llm::Tensor quantLogits = llm::Tensor::Zeros(llm::Shape{tokens.size(), config.vocabSize});
  ASSERT_TRUE(model.Forward(tokens, fp32Logits).IsOk());
  ASSERT_TRUE(quantModel.Value().Forward(tokens, quantLogits).IsOk());

  float maxDiff = 0.0f;
  for (llm::Index i = 0; i < static_cast<llm::Index>(fp32Logits.Numel()); ++i) {
    maxDiff = std::max(maxDiff, std::fabs(fp32Logits[i] - quantLogits[i]));
  }
  EXPECT_LT(maxDiff, 0.5f);

  std::remove(fp32Path.c_str());
  std::remove(quantPath.c_str());
}

TEST(QuantizationTest, RuntimeLoadForwardCloseToFp32) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerModel model(config);
  std::mt19937 rng(43);
  model.ResetParameters(rng);

  const std::string fp32Path = "quant_runtime_fp32.ckpt";
  const std::string quantPath = "quant_runtime_int8.ckptq";
  ASSERT_TRUE(llm::model::Checkpoint::Save(model, fp32Path).IsOk());
  ASSERT_TRUE(llm::quantization::QuantCheckpoint::ConvertFile(fp32Path, quantPath).IsOk());

  const auto runtimeModel = llm::quantization::QuantCheckpoint::LoadRuntime(quantPath);
  ASSERT_TRUE(runtimeModel.IsOk());

  const std::vector<llm::TokenId> tokens = {3, 4, 5};
  llm::Tensor fp32Logits = llm::Tensor::Zeros(llm::Shape{tokens.size(), config.vocabSize});
  llm::Tensor runtimeLogits = llm::Tensor::Zeros(llm::Shape{tokens.size(), config.vocabSize});
  ASSERT_TRUE(model.Forward(tokens, fp32Logits).IsOk());
  ASSERT_TRUE(runtimeModel.Value().Forward(tokens, runtimeLogits).IsOk());

  float maxDiff = 0.0f;
  for (llm::Index i = 0; i < static_cast<llm::Index>(fp32Logits.Numel()); ++i) {
    maxDiff = std::max(maxDiff, std::fabs(fp32Logits[i] - runtimeLogits[i]));
  }
  EXPECT_LT(maxDiff, 0.5f);

  std::remove(fp32Path.c_str());
  std::remove(quantPath.c_str());
}
