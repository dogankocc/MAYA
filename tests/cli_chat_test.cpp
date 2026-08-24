#include <cstdio>
#include <gtest/gtest.h>
#include <random>

#include "llm/cli/chat_session.hpp"
#include "llm/model/checkpoint/checkpoint.hpp"
#include "llm/model/transformer.hpp"
#include "llm/tokenizer/bpe_tokenizer.hpp"

TEST(ChatSessionTest, CompleteReturnsText) {
  llm::BpeTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.Train("hello world test", 64).IsOk());
  const std::size_t vocabSize = tokenizer.GetVocabulary().Size();

  llm::ModelConfig config;
  config.vocabSize = vocabSize;
  config.hiddenDim = 32;
  config.numLayers = 1;
  config.numHeads = 4;
  config.numKvHeads = 2;
  config.intermediateDim = 64;
  config.maxSeqLen = 32;

  llm::model::TransformerModel model(config);
  std::mt19937 rng(1);
  model.ResetParameters(rng);

  const std::string modelPath = "chat_test_model.ckpt";
  const std::string tokenizerPath = "chat_test_tokenizer";
  ASSERT_TRUE(llm::model::Checkpoint::Save(model, modelPath).IsOk());
  ASSERT_TRUE(tokenizer.Save(tokenizerPath).IsOk());

  llm::cli::ChatSession session;
  ASSERT_TRUE(session.Load(modelPath, tokenizerPath).IsOk());
  session.Sampling().greedy = true;

  const auto reply = session.Complete("hello", 4, rng);
  ASSERT_TRUE(reply.IsOk());

  std::remove(modelPath.c_str());
  std::remove((tokenizerPath + "/vocab.txt").c_str());
  std::remove((tokenizerPath + "/merges.txt").c_str());
}
