#include <cmath>
#include <gtest/gtest.h>
#include <random>

#include "llm/core/config.hpp"
#include "llm/model/attention.hpp"
#include "llm/model/ffn.hpp"
#include "llm/model/rms_norm.hpp"
#include "llm/model/rope.hpp"
#include "llm/model/transformer.hpp"
#include "llm/model/transformer_block.hpp"

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

TEST(ModelRmsNormTest, NormalizesRows) {
  llm::Tensor input = llm::Tensor::FromBuffer(llm::Shape{2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f});
  llm::Tensor weight = llm::Tensor::Ones(llm::Shape{4});
  llm::Tensor output = llm::Tensor::Zeros(llm::Shape{2, 4});

  ASSERT_TRUE(llm::model::RmsNorm(input, weight, 1e-5f, output).IsOk());

  float sumSquares = 0.0f;
  for (int i = 0; i < 4; ++i) {
    sumSquares += output[static_cast<llm::Index>(i)] * output[static_cast<llm::Index>(i)];
  }
  EXPECT_NEAR(sumSquares, 4.0f, 1e-3f);
}

TEST(ModelRopeTest, ChangesTensorValues) {
  llm::Tensor tensor = llm::Tensor::FromBuffer(llm::Shape{1, 8}, std::vector<llm::Scalar>(8, 1.0f));
  llm::model::RopeCache cache;
  cache.Build(4, 8, 10000.0f);

  const llm::Scalar before = tensor[0];
  llm::model::ApplyRope(tensor, cache, 2, 4, 1);
  EXPECT_NE(tensor[0], before);
}

TEST(ModelAttentionTest, ForwardProducesHiddenShape) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::MultiHeadAttention attention(config);
  std::mt19937 rng(42);
  attention.ResetParameters(rng);

  llm::model::RopeCache rope;
  rope.Build(config.hiddenDim / config.numHeads, config.maxSeqLen, config.ropeTheta);

  llm::Tensor input = llm::Tensor::Zeros(llm::Shape{3, config.hiddenDim});
  llm::Tensor output = llm::Tensor::Zeros(llm::Shape{3, config.hiddenDim});
  ASSERT_TRUE(attention.Forward(input, rope, output).IsOk());
  EXPECT_EQ(output.GetShape(), llm::Shape({3, config.hiddenDim}));
}

TEST(ModelFfnTest, ForwardProducesHiddenShape) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::SwiGluFfn ffn(config);
  std::mt19937 rng(7);
  ffn.ResetParameters(rng);

  llm::Tensor input = llm::Tensor::Zeros(llm::Shape{3, config.hiddenDim});
  llm::Tensor output = llm::Tensor::Zeros(llm::Shape{3, config.hiddenDim});
  ASSERT_TRUE(ffn.Forward(input, output).IsOk());
  EXPECT_EQ(output.GetShape(), llm::Shape({3, config.hiddenDim}));
}

TEST(ModelBlockTest, ForwardProducesSameShape) {
  const llm::ModelConfig config = SmallConfig();
  llm::model::TransformerBlock block(config);
  std::mt19937 rng(11);
  block.ResetParameters(rng);

  llm::model::RopeCache rope;
  rope.Build(config.hiddenDim / config.numHeads, config.maxSeqLen, config.ropeTheta);

  llm::Tensor input = llm::Tensor::Zeros(llm::Shape{3, config.hiddenDim});
  llm::Tensor output = llm::Tensor::Zeros(llm::Shape{3, config.hiddenDim});
  ASSERT_TRUE(block.Forward(input, rope, output).IsOk());
  EXPECT_EQ(output.GetShape(), llm::Shape({3, config.hiddenDim}));
}

TEST(ModelTransformerTest, ForwardProducesLogits) {
  llm::ModelConfig config = SmallConfig();
  ASSERT_TRUE(config.Validate().IsOk());

  llm::model::TransformerModel model(config);
  std::mt19937 rng(99);
  model.ResetParameters(rng);

  const std::vector<llm::TokenId> tokens = {5, 6, 7};
  llm::Tensor logits = llm::Tensor::Zeros(llm::Shape{tokens.size(), config.vocabSize});
  ASSERT_TRUE(model.Forward(tokens, logits).IsOk());
  EXPECT_EQ(logits.GetShape(), llm::Shape({3, config.vocabSize}));
}
