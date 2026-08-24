#include <cstdio>
#include <gtest/gtest.h>

#include "llm/tokenizer/vocabulary.hpp"

TEST(VocabularyTest, SpecialTokensHaveFixedIds) {
  const llm::Vocabulary vocabulary;
  EXPECT_EQ(vocabulary.GetId(llm::kPadToken), llm::kPadTokenId);
  EXPECT_EQ(vocabulary.GetId(llm::kUnkToken), llm::kUnkTokenId);
  EXPECT_EQ(vocabulary.GetId(llm::kBosToken), llm::kBosTokenId);
  EXPECT_EQ(vocabulary.GetId(llm::kEosToken), llm::kEosTokenId);
}

TEST(VocabularyTest, UnknownTokenMapsToUnk) {
  const llm::Vocabulary vocabulary;
  EXPECT_EQ(vocabulary.GetId("does-not-exist"), llm::kUnkTokenId);
}

TEST(VocabularyTest, SaveAndLoadRoundTrip) {
  llm::Vocabulary vocabulary;
  vocabulary.AddToken("a");
  vocabulary.AddToken("b");

  const std::string path = "test_vocab.txt";
  ASSERT_TRUE(vocabulary.Save(path).IsOk());

  const auto loaded = llm::Vocabulary::Load(path);
  ASSERT_TRUE(loaded.IsOk());
  EXPECT_EQ(loaded.Value().GetId("a"), vocabulary.GetId("a"));
  EXPECT_EQ(loaded.Value().GetToken(vocabulary.GetId("b")), "b");

  std::remove(path.c_str());
}
