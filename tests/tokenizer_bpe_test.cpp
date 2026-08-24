#include <cstdio>
#include <gtest/gtest.h>

#include "llm/tokenizer/bpe_tokenizer.hpp"

TEST(BpeTokenizerTest, TrainBuildsMerges) {
  llm::BpeTokenizer tokenizer;
  const std::string corpus = "araba araba araba araba araba araba";
  ASSERT_TRUE(tokenizer.Train(corpus, 32).IsOk());
  EXPECT_GT(tokenizer.GetVocabulary().Size(), llm::kSpecialTokenCount);
}

TEST(BpeTokenizerTest, EncodeDecodeRoundTrip) {
  llm::BpeTokenizer tokenizer;
  const std::string corpus =
      "araba araba araba test test test model model model model";
  ASSERT_TRUE(tokenizer.Train(corpus, 64).IsOk());

  const std::string text = "araba";
  const std::vector<llm::TokenId> ids = tokenizer.Encode(text);
  EXPECT_FALSE(ids.empty());

  const std::string decoded = tokenizer.Decode(ids);
  EXPECT_NE(decoded.find("araba"), std::string::npos);
}

TEST(BpeTokenizerTest, EncodeWithSpecialTokens) {
  llm::BpeTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.Train("hello hello hello", 32).IsOk());

  const auto ids = tokenizer.EncodeWithSpecialTokens("hello", true, true);
  ASSERT_GE(ids.size(), 3U);
  EXPECT_EQ(ids.front(), llm::kBosTokenId);
  EXPECT_EQ(ids.back(), llm::kEosTokenId);
}

TEST(BpeTokenizerTest, SaveAndLoadRoundTrip) {
  llm::BpeTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.Train("merhaba dunya merhaba dunya", 48).IsOk());

  const std::string directory = "test_tokenizer";
  ASSERT_TRUE(tokenizer.Save(directory).IsOk());

  const auto loaded = llm::BpeTokenizer::Load(directory);
  ASSERT_TRUE(loaded.IsOk());

  const std::string text = "merhaba";
  const auto& loadedTokenizer = loaded.Value();
  const std::string decoded = loadedTokenizer.Decode(loadedTokenizer.Encode(text));
  EXPECT_NE(decoded.find("merhaba"), std::string::npos);

  std::remove((directory + "/vocab.txt").c_str());
  std::remove((directory + "/merges.txt").c_str());
#ifdef _WIN32
  _rmdir(directory.c_str());
#else
  rmdir(directory.c_str());
#endif
}
